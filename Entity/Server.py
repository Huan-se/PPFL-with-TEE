import torch
import torch.nn as nn
import copy
import numpy as np
import os
import csv
from collections import defaultdict

# [移除] 不再使用 SuperBitLSH，改为 TEE 流式投影
# from _utils_.LSH_proj_extra import SuperBitLSH 

from Defence.score import ScoreCalculator
from Defence.kickout import KickoutManager
from Defence.layers_proj_detect import Layers_Proj_Detector
from _utils_.tee_adapter import TEEAdapter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义大素数 (必须与 C++ Enclave 保持一致)
MOD = 9223372036854775783
SCALE = 10000.0

class Server(object):
    def __init__(self, model_class, test_dataloader, device_str, detection_method="none", defense_config=None, seed=42, verbose=False, log_file_path=None, malicious_clients=None):
        self.device = torch.device(device_str)
        self.global_model = model_class().to(self.device)
        self.test_dataloader = test_dataloader
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.tee_adapter = TEEAdapter()

        self.detection_method = detection_method
        self.verbose = verbose
        self.log_file_path = log_file_path
        self.seed = seed # 保存全局种子用于计算 proj_seed
        
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
        
        # 记录上一轮的全局模型参数，用于计算全局更新方向
        self.w_old_global_flat = self._flatten_params(self.global_model)

        should_log = any(k in self.detection_method for k in ["mesas", "projected", "layers_proj"])
        if self.log_file_path and should_log:
            self._init_log_file()

    def _init_log_file(self):
        log_dir = os.path.dirname(self.log_file_path)
        if log_dir: os.makedirs(log_dir, exist_ok=True)
        headers = ["Round", "Client_ID", "Type", "Score", "Status"]
        target_layers = self.defense_config.get('target_layers', [])
        scopes = ['full'] + [f'layer_{name}' for name in target_layers]
        metrics = ['l2', 'var', 'dist']
        try:
            with open(self.log_file_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                for scope in scopes:
                    for metric in metrics:
                        headers.extend([f"{scope}_{metric}", f"{scope}_{metric}_threshold"])
                writer.writerow(headers)
        except Exception: pass
        
    def _flatten_params(self, model):
        params = []
        for param in model.parameters():
            params.append(param.data.view(-1).cpu().numpy())
        return np.concatenate(params).astype(np.float32)

    def get_global_params_and_proj(self):
        """[Phase 1] 下发全局参数 (投影矩阵由 Seed 在 TEE 内流式生成，无需下发文件)"""
        # 更新 w_old 为当前的全局模型
        self.w_old_global_flat = self._flatten_params(self.global_model)
        return copy.deepcopy(self.global_model.state_dict()), None

    def calculate_weights(self, client_id_list, client_features_dict_list, client_data_sizes, current_round=0):
        client_projections = {cid: feat for cid, feat in zip(client_id_list, client_features_dict_list)}
        
        # [核心修复] 使用 TEE 接口流式投影全局模型更新，作为检测基准
        self._update_global_direction_feature(current_round)
        
        weights = {}

        if any(k in self.detection_method for k in ["mesas", "projected", "layers_proj"]):
            if self.verbose: print(f"  [Server] Executing {self.detection_method} detection (Round {current_round})...")
            
            raw_weights, logs, global_stats = self.mesas_detector.detect(
                client_projections, 
                self.global_update_direction, # 传入对齐后的全局方向
                self.suspect_counters, 
                verbose=self.verbose
            )
            
            if self.log_file_path: self._write_detection_log(current_round, logs, raw_weights, global_stats)
            for cid in sorted(logs.keys()):
                status = logs[cid].get('status', 'NORMAL')
                if "SUSPECT" in status: self.detection_history[cid]['suspect_cnt'] += 1
                if "KICKED" in status: self.detection_history[cid]['kicked_cnt'] += 1
            total_score = sum(raw_weights.values())
            weights = {cid: s / total_score for cid, s in raw_weights.items()} if total_score > 0 else {cid: 0.0 for cid in raw_weights}
            
        else:
            full_features = [f['full'] for f in client_features_dict_list]
            weights = self._fallback_old_detection(client_id_list, full_features, client_data_sizes)
        self.current_round_weights = weights
        return weights

    # =========================================================================
    # 安全聚合核心逻辑
    # =========================================================================
    def secure_aggregation(self, client_objects, active_ids, round_num):
        if self.verbose: print(f"\n[Server] Starting Secure Aggregation (Round {round_num})...")
        weights_map = self.current_round_weights
        
        encrypted_grads = {}
        online_clients = []
        total_params_len = sum(p.numel() for p in self.global_model.parameters())

        # 1. 收集密文
        for client in client_objects:
            if client.client_id not in active_ids: continue
            w = weights_map.get(client.client_id, 0.0)
            if w <= 1e-9: continue
            try:
                c_grad = client.tee_step2_upload(w, active_ids, self.seed_mask_root, self.seed_global_0)
                encrypted_grads[client.client_id] = c_grad
                online_clients.append(client.client_id)
            except Exception as e:
                print(f"  [Warning] Client {client.client_id} failed to upload: {e}")

        if not online_clients:
            print("  [Error] No clients uploaded gradients.")
            return

        # 2. 掉线恢复
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
                except Exception as e:
                    print(f"  [Warning] Failed to get share from Client {client.client_id}: {e}")

        if len(shares_collected) < threshold:
            print(f"  [Error] Insufficient shares ({len(shares_collected)} < {threshold}).")
            return
        
        # 3. 重构秘密
        secret_vector = self._reconstruct_secrets(shares_collected, threshold)
        if secret_vector is None:
            print("  [Error] Failed to reconstruct secrets.")
            return

        delta = secret_vector[0]
        alpha_seed = secret_vector[1]
        beta_seeds = secret_vector[2:]

        # 4. 消除掩码
        if self.verbose: print("  [Server] Unmasking and aggregating...")

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

        coeff_M = (1 - delta) % MOD
        term_M = (coeff_M * vec_M) % MOD
        
        result_int = (agg_cipher_obj - term_M - vec_B_sum) % MOD
        result_int = (result_int + MOD) % MOD
        
        # 5. 反定点化
        threshold_neg = MOD // 2
        mask_neg = result_int > threshold_neg
        temp_arr = np.array(result_int, dtype=object)
        temp_arr[mask_neg] -= MOD 
        result_float = temp_arr.astype(np.float32) / SCALE
        
        if np.isnan(result_float).any() or np.isinf(result_float).any():
            print("\n  [CRITICAL ERROR] NaN or Inf detected in aggregated gradients!")
            return 

        self._apply_global_update(result_float)
        self.seed_global_0 = (self.seed_global_0 + 1) & 0x7FFFFFFF

        if self.verbose: print("  [Server] Secure aggregation completed successfully.")

    def _update_global_direction_feature(self, current_round):
        """
        [新增] 使用 TEE Adapter 生成全局模型的流式投影特征
        目的: 确保 Server 端使用的基准特征与 Client 端生成的特征在数学上完全一致 (Seed -> Gaussian)
        """
        try:
            # 1. 扁平化当前参数 (w_new)
            w_new_flat = self._flatten_params(self.global_model)
            
            # 2. 准备 w_old (上一轮的 global model)
            if self.w_old_global_flat is None:
                self.w_old_global_flat = np.zeros_like(w_new_flat)
                
            # 3. 计算本轮种子 (需与 Client 保持一致)
            proj_seed = int(self.seed + current_round)
            
            # 4. 调用 TEE Adapter 进行投影
            # 使用特殊 ID (如 -1) 表示 Server
            projection = self.tee_adapter.prepare_gradient(
                -1, 
                proj_seed, 
                w_new_flat, 
                self.w_old_global_flat
            )
            
            # 5. 存储结果
            self.global_update_direction = {
                'full': projection,
                'layers': {} # 如果需要分层检测，需在 Adapter 增加分层返回逻辑，此处暂略
            }
            
        except Exception as e:
            print(f"  [Warning] Global direction projection failed: {e}")
            self.global_update_direction = None

    def _reconstruct_secrets(self, shares, threshold):
        if len(shares) < threshold: return None
        selected_shares = shares[:threshold]
        vec_len = len(selected_shares[0]['v'])
        reconstructed = np.zeros(vec_len, dtype=np.int64)
        x_s = [s['x'] for s in selected_shares]
        for i in range(vec_len):
            y_s = [s['v'][i] for s in selected_shares]
            reconstructed[i] = self._lagrange_interpolate_zero(x_s, y_s)
        return reconstructed

    def _lagrange_interpolate_zero(self, x_s, y_s):
        total = 0
        k = len(x_s)
        for j in range(k):
            numerator = 1
            denominator = 1
            for m in range(k):
                if m == j: continue
                numerator = (numerator * (0 - x_s[m])) % MOD
                denominator = (denominator * (x_s[j] - x_s[m])) % MOD
            inv_den = pow(int(denominator), -1, int(MOD))
            lj_0 = (numerator * inv_den) % MOD
            term = (y_s[j] * lj_0) % MOD
            total = (total + term) % MOD
        return (total + MOD) % MOD

    def _apply_global_update(self, update_flat):
        idx = 0
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                numel = param.numel()
                grad_segment = update_flat[idx : idx+numel]
                grad_tensor = torch.from_numpy(grad_segment).view(param.shape).to(self.device)
                param.data.add_(grad_tensor)
                idx += numel

    def _fallback_old_detection(self, ids, f, s):
        return {cid: 1.0/len(ids) for cid in ids}
    def _write_detection_log(self, r, l, w, s): pass 
    def evaluate(self):
        self.global_model.eval()
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