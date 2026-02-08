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
from _utils_.tee_adapter import get_tee_adapter_singleton
from _utils_.server_adapter import ServerAdapter # [新增]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MOD = 9223372036854775783
SCALE = 100000000.0 # [同步修改] 与 Enclave 保持一致

class Server(object):
    def __init__(self, model_class, test_dataloader, device_str, detection_method="none", defense_config=None, seed=42, verbose=False, log_file_path=None, malicious_clients=None):
        self.device = torch.device(device_str)
        self.global_model = model_class().to(self.device)
        self.test_dataloader = test_dataloader
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        
        self.tee_adapter = get_tee_adapter_singleton()
        self.server_adapter = ServerAdapter() # [新增] C++ Server Core

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
        if self.log_file_path: self._init_log_file()

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
            # Log & Weights logic...
            total_score = sum(raw_weights.values())
            weights = {cid: s / total_score for cid, s in raw_weights.items()} if total_score > 0 else {cid: 0.0 for cid in raw_weights}
        else:
            weights = {cid: 1.0/len(client_id_list) for cid in client_id_list}
        self.current_round_weights = weights
        return weights

    def _write_detection_log(self, round_num, logs, weights, global_stats):
        if not self.log_file_path: return
        try:
            with open(self.log_file_path, 'a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                for cid in sorted(logs.keys()):
                    info = logs[cid]
                    score = weights.get(cid, 0.0)
                    status = info.get('status', 'NORMAL')
                    row = [round_num, cid, "Client", f"{score:.4f}", status]
                    # 补充 metric 详情... (这里简化处理，只写基本信息)
                    writer.writerow(row)
        except Exception: pass

    def secure_aggregation(self, client_objects, active_ids, round_num):
        print(f"\n[Server] >>> STARTING V4 SECURE AGGREGATION (ROUND {round_num}) [C++ REE] <<<")
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
        
        # --- [关键步骤 1] 使用 C++ 计算 Delta ---
        t_recon_start = time.time()
        delta, n_sum = self.server_adapter.calculate_secrets(
            self.seed_mask_root, active_ids, u2_ids
        )
        t_recon_end = time.time()
        
        # --- [关键步骤 2] 密文聚合 ---
        t_agg_start = time.time()
        agg_cipher_obj = np.zeros(total_params_len, dtype=object)
        for cid in u2_ids:
            agg_cipher_obj += encrypted_grads[cid].astype(object)
        agg_cipher_obj %= MOD

        # --- [关键步骤 3] 使用 C++ 生成噪声向量 ---
        noise_vector = self.server_adapter.generate_noise_vector(
            self.seed_mask_root, 
            self.seed_global_0, 
            delta, 
            u2_ids, 
            total_params_len
        )
        
        # --- [关键步骤 4] 消除掩码 ---
        noise_obj = noise_vector.astype(object)
        result_int = (agg_cipher_obj - noise_obj) % MOD
        result_int = (result_int + MOD) % MOD 
        t_agg_end = time.time()
        
        # 反量化
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
        
        print(f"  Time Breakdown: Collect={t_collect_end-t_collect_start:.2f}s, Recon={t_recon_end-t_recon_start:.2f}s, Agg+Noise={t_agg_end-t_agg_start:.2f}s")

    def _update_global_direction_feature(self, current_round):
        try:
            w_new_flat = self._flatten_params(self.global_model)
            if self.w_old_global_flat is None:
                self.w_old_global_flat = np.zeros_like(w_new_flat)
            proj_seed = int(self.seed + current_round)
            projection, _ = self.tee_adapter.prepare_gradient(
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
        self.global_model.train() 
        for m in self.global_model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.track_running_stats = False
        test_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for inputs, targets in self.test_dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.global_model(inputs)
                test_loss += self.criterion(outputs, targets).item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        for m in self.global_model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.track_running_stats = True
        return 100.*correct/total, test_loss/len(self.test_dataloader)

    def evaluate_asr(self, loader, poison_loader):
        self.global_model.train() 
        for m in self.global_model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.track_running_stats = False
        correct = 0
        total = 0
        target_class = None
        params = poison_loader.attack_params
        if "backdoor" in poison_loader.attack_methods:
            target_class = params.get("backdoor_target", 0) 
        elif "label_flip" in poison_loader.attack_methods:
            target_class = params.get("target_class", 7)    
        if target_class is None: return 0.0
        
        with torch.no_grad():
            for data, target in loader:
                non_target_indices = torch.where(target != target_class)[0]
                if len(non_target_indices) == 0: continue
                data_subset = data[non_target_indices]
                target_subset = target[non_target_indices]
                data_poisoned, target_poisoned = poison_loader.apply_data_poison(data_subset, target_subset)
                data_poisoned = data_poisoned.to(self.device)
                target_poisoned = target_poisoned.to(self.device)
                output = self.global_model(data_poisoned)
                _, predicted = output.max(1)
                total += len(target_poisoned)
                correct += predicted.eq(target_poisoned).sum().item()
        for m in self.global_model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.track_running_stats = True
        if total == 0: return 0.0
        return 100. * correct / total