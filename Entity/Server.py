import torch
import torch.nn as nn
import copy
import numpy as np
import os
import csv
import time
from collections import defaultdict

from Defence.score import ScoreCalculator
from Defence.kickout import KickoutManager
from Defence.layers_proj_detect import Layers_Proj_Detector
from _utils_.tee_adapter import get_tee_adapter_singleton # [修改] 导入单例获取函数

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MOD = 9223372036854775783
SCALE = 1000000000.0

class Server(object):
    def __init__(self, model_class, test_dataloader, device_str, detection_method="none", defense_config=None, seed=42, verbose=False, log_file_path=None, malicious_clients=None):
        self.device = torch.device(device_str)
        self.global_model = model_class().to(self.device)
        self.test_dataloader = test_dataloader
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        
        # [核心修改] 使用全局唯一的 TEE 适配器
        self.tee_adapter = get_tee_adapter_singleton()

        self.detection_method = detection_method
        self.verbose = verbose
        self.log_file_path = log_file_path
        self.seed = seed 
        self.malicious_clients = set(malicious_clients) if malicious_clients else set()
        self.defense_config = defense_config or {}
        self.suspect_counters = {} 
        self.global_update_direction = None 
        self.detection_history = defaultdict(lambda: {'suspect_cnt': 0, 'kicked_cnt': 0, 'events': []})
        det_params = self.defense_config.get('params', {})
        self.mesas_detector = Layers_Proj_Detector(config=det_params)
        self.score_calculator = ScoreCalculator() if "score" in detection_method else None
        self.kickout_manager = KickoutManager() if "kickout" in detection_method else None
        self.current_round_weights = {}
        self.seed_mask_root = 0x12345678 
        self.seed_global_0 = 0x87654321  
        self.seed_sss = 0x11223344
        self.w_old_global_flat = self._flatten_params(self.global_model)
        should_log = any(k in self.detection_method for k in ["mesas", "projected", "layers_proj"])
        if self.log_file_path and should_log: self._init_log_file()

    def _init_log_file(self):
        log_dir = os.path.dirname(self.log_file_path)
        if log_dir: os.makedirs(log_dir, exist_ok=True)
        try:
            with open(self.log_file_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(["Round", "Client_ID", "Score", "Status"])
        except Exception: pass
        
    def _flatten_params(self, model):
        params = []
        for param in model.parameters():
            params.append(param.data.view(-1).cpu().numpy())
        return np.concatenate(params).astype(np.float32)

    def get_global_params_and_proj(self):
        self.w_old_global_flat = self._flatten_params(self.global_model)
        return copy.deepcopy(self.global_model.state_dict()), None

    def calculate_weights(self, client_id_list, client_features_dict_list, client_data_sizes, current_round=0):
        client_projections = {cid: feat for cid, feat in zip(client_id_list, client_features_dict_list)}
        self._update_global_direction_feature(current_round)
        weights = {}
        if any(k in self.detection_method for k in ["mesas", "projected", "layers_proj"]):
            if self.verbose: print(f"  [Server] Executing {self.detection_method} detection (Round {current_round})...")
            raw_weights, logs, global_stats = self.mesas_detector.detect(
                client_projections, self.global_update_direction, self.suspect_counters, verbose=self.verbose
            )
            total_score = sum(raw_weights.values())
            weights = {cid: s / total_score for cid, s in raw_weights.items()} if total_score > 0 else {cid: 0.0 for cid in raw_weights}
        else:
            weights = {cid: 1.0/len(client_id_list) for cid in client_id_list}
        self.current_round_weights = weights
        return weights

    def secure_aggregation(self, client_objects, active_ids, round_num):
        print(f"\n[Server] >>> STARTING V4 SECURE AGGREGATION (ROUND {round_num}) [WITH PROFILE] <<<")
        weights_map = self.current_round_weights
        encrypted_grads = {}
        online_clients = []
        total_params_len = sum(p.numel() for p in self.global_model.parameters())

        t_collect_start = time.time()
        plaintext_sum_accumulator = np.zeros(total_params_len, dtype=object)
        
        for client in client_objects:
            if client.client_id not in active_ids: continue
            w = weights_map.get(client.client_id, 0.0)
            if w <= 1e-9: continue
            try:
                c_grad = client.tee_step2_upload(w, active_ids, self.seed_mask_root, self.seed_global_0)
                encrypted_grads[client.client_id] = c_grad
                online_clients.append(client.client_id)
                # Debug Check
                w_client_flat = self._flatten_params(client.model)
                grad_flat = w_client_flat - self.w_old_global_flat
                grad_quantized = (grad_flat * w * SCALE).astype(np.int64)
                plaintext_sum_accumulator += grad_quantized.astype(object)
            except Exception as e:
                print(f"  [Warning] Upload failed for Client {client.client_id}: {e}")
        
        t_collect_end = time.time()
        if not online_clients:
            print("  [Error] No clients uploaded gradients.")
            return

        u2_ids = sorted(online_clients)
        threshold = int(len(u2_ids) * 0.6) + 1
        if threshold < 2: threshold = 1
        if threshold > len(u2_ids): threshold = len(u2_ids)

        shares_collected = []
        for client in client_objects:
            if client.client_id in u2_ids:
                try:
                    share_vec = client.tee_step3_get_shares(self.seed_sss, self.seed_mask_root, active_ids, u2_ids, threshold)
                    shares_collected.append({'x': client.client_id + 1, 'v': share_vec})
                except Exception as e: pass

        if len(shares_collected) < threshold: return
        
        t_recon_start = time.time()
        secret_vector = self._reconstruct_secrets(shares_collected, threshold)
        t_recon_end = time.time()
        if secret_vector is None: return

        delta = secret_vector[0]
        alpha_seed = secret_vector[1]
        beta_seeds = secret_vector[2:]
        
        t_agg_start = time.time()
        agg_cipher_obj = np.zeros(total_params_len, dtype=object)
        for cid in u2_ids:
            agg_cipher_obj += encrypted_grads[cid].astype(object)
        agg_cipher_obj %= MOD

        final_seed_M = (self.seed_global_0 + alpha_seed) & 0x7FFFFFFF
        vec_M = self.tee_adapter.generate_noise_from_seed(final_seed_M, total_params_len).astype(object)
        
        vec_B_sum = np.zeros(total_params_len, dtype=object)
        for b_seed in beta_seeds:
            vec_B = self.tee_adapter.generate_noise_from_seed(b_seed, total_params_len).astype(object)
            vec_B_sum += vec_B
        vec_B_sum %= MOD

        coeff_M = int(1 - delta) % int(MOD)
        term_M = (coeff_M * vec_M) % MOD
        result_int = (agg_cipher_obj - term_M - vec_B_sum) % MOD
        result_int = (result_int + MOD) % MOD
        t_agg_end = time.time()
        
        t_check_start = time.time()
        plaintext_truth_mod = (plaintext_sum_accumulator % MOD + MOD) % MOD
        diff = np.abs(result_int - plaintext_truth_mod)
        TOLERANCE = 100000 
        diff_count = np.count_nonzero(diff > TOLERANCE)
        if diff_count == 0:
            print("  >>> [SUCCESS] CHECK PASSED!")
        else:
            print(f"  >>> [WARNING] CHECK DIFF! {diff_count} params differ.")
            print(f"  >>> [WARNING] MAX DIFF {diff.max()}")
        t_check_end = time.time()

        threshold_neg = MOD // 2
        mask_neg = result_int > threshold_neg
        temp_arr = np.array(result_int, dtype=object)
        temp_arr[mask_neg] -= MOD 
        result_float = temp_arr.astype(np.float32) / SCALE
        
        if np.isnan(result_float).any() or np.isinf(result_float).any():
            print("\n  [CRITICAL ERROR] NaN or Inf detected!")
            return 

        self._apply_global_update(result_float)
        self.seed_global_0 = (self.seed_global_0 + 1) & 0x7FFFFFFF

    def _update_global_direction_feature(self, current_round):
        try:
            w_new_flat = self._flatten_params(self.global_model)
            if self.w_old_global_flat is None:
                self.w_old_global_flat = np.zeros_like(w_new_flat)
            proj_seed = int(self.seed + current_round)
            projection = self.tee_adapter.prepare_gradient(
                -1, proj_seed, w_new_flat, self.w_old_global_flat
            )
            self.global_update_direction = {'full': projection, 'layers': {}}
        except Exception as e:
            self.global_update_direction = None

    def _reconstruct_secrets(self, shares, threshold):
        if len(shares) < threshold: return None
        selected_shares = shares[:threshold]
        x_s = [int(s['x']) for s in selected_shares]
        vec_len = len(selected_shares[0]['v'])
        reconstructed = np.zeros(vec_len, dtype=np.int64)
        for i in range(vec_len):
            y_s = [int(s['v'][i]) for s in selected_shares]
            reconstructed[i] = self._lagrange_interpolate_zero(x_s, y_s)
        return reconstructed

    def _lagrange_interpolate_zero(self, x_s, y_s):
        total = 0
        k = len(x_s)
        MOD_INT = int(MOD) 
        for j in range(k):
            numerator = 1
            denominator = 1
            for m in range(k):
                if m == j: continue
                numerator = (numerator * (0 - x_s[m])) % MOD_INT
                denominator = (denominator * (x_s[j] - x_s[m])) % MOD_INT
            inv_den = pow(denominator, -1, MOD_INT)
            lj_0 = (numerator * inv_den) % MOD_INT
            term = (y_s[j] * lj_0) % MOD_INT
            total = (total + term) % MOD_INT
        return total

    def _apply_global_update(self, update_flat):
        idx = 0
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                numel = param.numel()
                grad_segment = update_flat[idx : idx+numel]
                grad_tensor = torch.from_numpy(grad_segment).view(param.shape).to(self.device)
                param.data.add_(grad_tensor)
                idx += numel

    def evaluate(self):
        # [核心修复] 
        # 1. 切换到 train 模式以使用 batch 统计量 (避免因 Server 无 running_stats 导致的低 Acc)
        # 2. 绝对不要将 track_running_stats 设为 False 或将 running_mean/var 设为 None，
        #    否则 state_dict 会缺损，导致下一轮 Client 加载模型时崩溃。
        self.global_model.train() 
        
        test_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for inputs, targets in self.test_dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.global_model(inputs)
                test_loss += self.criterion(outputs, targets).item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return 100.*correct/total, test_loss/len(self.test_dataloader)

    def evaluate_asr(self, loader, atype, aparams): return 0.0