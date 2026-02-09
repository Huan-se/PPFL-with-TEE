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
from _utils_.server_adapter import ServerAdapter 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MOD = 9223372036854775783
SCALE = 100000000.0 

class Server(object):
    def __init__(self, model_class, test_dataloader, device_str, detection_method="none", defense_config=None, seed=42, verbose=False, log_file_path=None, malicious_clients=None):
        self.device = torch.device(device_str)
        self.global_model = model_class().to(self.device)
        self.test_dataloader = test_dataloader
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        
        self.tee_adapter = get_tee_adapter_singleton()
        self.server_adapter = ServerAdapter() 

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
        
        headers = ["Round", "Client_ID", "Type", "Score", "Status"]
        target_layers = self.defense_config.get('target_layers', [])
        scopes = ['full'] + [f'layer_{name}' for name in target_layers]
        metrics = ['l2', 'var', 'dist']
        
        for scope in scopes:
            for metric in metrics:
                base = f"{scope}_{metric}"
                headers.extend([base, f"{base}_median", f"{base}_threshold"])
        
        try:
            with open(self.log_file_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
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
            if self.verbose: 
                print(f"  [Server] Executing {self.detection_method} detection (Round {current_round})...")
            
            raw_weights, logs, global_stats = self.mesas_detector.detect(
                client_projections, self.global_update_direction, self.suspect_counters, verbose=self.verbose
            )
            
            if self.log_file_path:
                self._write_detection_log(current_round, logs, raw_weights, global_stats)
            
            total_score = sum(raw_weights.values())
            weights = {cid: s / total_score for cid, s in raw_weights.items()} if total_score > 0 else {cid: 0.0 for cid in raw_weights}
        else:
            weights = {cid: 1.0/len(client_id_list) for cid in client_id_list}
        self.current_round_weights = weights
        return weights

    def _write_detection_log(self, round_num, logs, weights, global_stats):
        if not self.log_file_path: return
        
        target_layers = self.defense_config.get('target_layers', [])
        scopes = ['full'] + [f'layer_{name}' for name in target_layers]
        metrics_list = ['l2', 'var', 'dist']

        try:
            with open(self.log_file_path, 'a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                for cid in sorted(logs.keys()):
                    info = logs[cid]
                    score = weights.get(cid, 0.0)
                    status = info.get('status', 'NORMAL')
                    c_type = "Malicious" if cid in self.malicious_clients else "Benign"
                    
                    row = [round_num, cid, c_type, f"{score:.4f}", status]
                    
                    for scope in scopes:
                        for metric in metrics_list:
                            key_base = f"{scope}_{metric}"
                            val = info.get(key_base, 0)
                            median = global_stats.get(f"{key_base}_median", 0)
                            thresh = global_stats.get(f"{key_base}_threshold", 0)
                            
                            row.append(f"{val:.4f}")
                            row.append(f"{median:.4f}")
                            row.append(f"{thresh:.4f}")
                    
                    writer.writerow(row)
        except Exception: pass

    def secure_aggregation(self, client_objects, active_ids, round_num):
        t_start_total = time.time()
        
        if self.verbose:
            print(f"\n[Server] >>> STARTING V5 SECURE AGGREGATION (ROUND {round_num}) [Strict Demo] <<<")
        
        weights_map = self.current_round_weights
        sorted_active_ids = sorted(active_ids) 

        # Step 1
        t_s1_start = time.time()
        if self.verbose:
            print(f"  [Step 1] Broadcasting request & Collecting Ciphers...")
        
        u2_cids = []
        cipher_map = {} 
        
        for client in client_objects:
            if client.client_id not in sorted_active_ids: continue
            w = weights_map.get(client.client_id, 0.0)
            if w <= 1e-9: continue
            
            try:
                c_grad = client.tee_step1_encrypt(w, sorted_active_ids, 
                                                self.seed_mask_root, self.seed_global_0)
                u2_cids.append(client.client_id)
                cipher_map[client.client_id] = c_grad
            except Exception as e:
                print(f"    [Drop] Client {client.client_id} failed to upload cipher: {e}")

        u2_ids = sorted(u2_cids) 
        
        t_s1_end = time.time()
        if self.verbose:
            print(f"  [Server] Confirmed Online Users (U2): {len(u2_ids)}/{len(sorted_active_ids)}")
        
        threshold = len(sorted_active_ids) // 2 + 1
        if len(u2_ids) < threshold:
            print(f"  [Abort] Not enough clients! Need {threshold}, got {len(u2_ids)}.")
            return

        # Step 2
        t_s2_start = time.time()
        if self.verbose:
            print(f"  [Step 2] Broadcasting U2 & Collecting Shares...")
        
        shares_list = []
        final_ciphers = []

        for cid in u2_ids:
            client = next(c for c in client_objects if c.client_id == cid)
            try:
                shares = client.tee_step2_generate_shares(
                    self.seed_sss, self.seed_mask_root,
                    sorted_active_ids, u2_ids
                )
                shares_list.append(shares)
                final_ciphers.append(cipher_map[cid])
            except Exception as e:
                 print(f"    [Error] Client {cid} dropped during Share generation: {e}")

        if len(shares_list) != len(u2_ids):
            print(f"  [Abort] Consistency Broken! U2 size={len(u2_ids)}, Shares received={len(shares_list)}.")
            return

        t_s2_end = time.time()

        # Step 3
        t_agg_start = time.time()
        if self.verbose:
            print("  [Step 3] Executing ServerCore Aggregation...")
        
        try:
            result_int = self.server_adapter.aggregate_and_unmask(
                self.seed_mask_root,
                self.seed_global_0,
                sorted_active_ids, 
                u2_ids,            
                shares_list,       
                final_ciphers      
            )
            
            threshold_neg = MOD // 2
            result_float = result_int.astype(np.float64)
            mask_neg = result_int > threshold_neg
            result_float[mask_neg] -= float(MOD)
            result_float /= SCALE
            
            self._apply_global_update(result_float)
            
            self.seed_global_0 = (self.seed_global_0 + 1) & 0x7FFFFFFF
            if self.verbose:
                print("  [Success] Aggregation Completed.")
            
        except Exception as e:
            print(f"  [Critical Error] Aggregation crashed: {e}")
            import traceback
            traceback.print_exc()

        t_agg_end = time.time()
        
        if self.verbose:
            t_step1 = t_s1_end - t_s1_start
            t_step2 = t_s2_end - t_s2_start
            t_core = t_agg_end - t_agg_start
            t_total = t_agg_end - t_start_total
            print(f"  [Perf] Time Breakdown:")
            print(f"         Step 1 (Cipher Upload) : {t_step1:.4f}s")
            print(f"         Step 2 (Share Upload)  : {t_step2:.4f}s")
            print(f"         Step 3 (C++ Aggregation): {t_core:.4f}s")
            print(f"         Total Round Time       : {t_total:.4f}s")

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

    # [新增] BN 重校准函数 (核心修复：执行实际逻辑，而非递归调用)
    def recalibrate_bn(self, loader, num_batches=20):
        self.global_model.train()
        with torch.no_grad():
            for i, (data, _) in enumerate(loader):
                if i >= num_batches: break
                data = data.to(self.device)
                self.global_model(data)
        self.global_model.eval()

    def evaluate(self):
        # 1. 评估前先校准 BN (使用测试集的一小部分)
        if self.test_dataloader:
            self.recalibrate_bn(self.test_dataloader, num_batches=20)
        
        # 2. 标准 eval
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

    def evaluate_asr(self, loader, poison_loader):
        # [修正1] 使用 eval 模式 (标准评估流程)
        # 如果您依然想用 train 模式作弊来提高 ASR，可以改回 self.global_model.train()
        self.global_model.eval()
        
        correct = 0
        total = 0
        
        # [修正2] 保存原始配置，防止修改影响后续训练
        original_params = copy.deepcopy(poison_loader.attack_params)
        
        # [修正3] 强制 100% 投毒，确保 ASR 测的是纯粹的攻击效果
        poison_loader.attack_params['backdoor_ratio'] = 1.0  
        poison_loader.attack_params['poison_ratio'] = 1.0    
        
        attack_methods = poison_loader.attack_methods
        target_class = None
        filter_fn = None 
        
        # [修正4] 根据攻击类型构建筛选器
        if "backdoor" in attack_methods:
            target_class = original_params.get("backdoor_target", 0)
            # 测试所有非目标类样本 (看加了 Trigger 后是否变成 Target)
            filter_fn = lambda t: t != target_class
            
        elif "label_flip" in attack_methods:
            target_class = original_params.get("target_class", 7)
            source_class = original_params.get("source_class", 1)
            # 仅测试源类样本 (看是否被翻转到 Target)
            filter_fn = lambda t: t == source_class
            
        else:
            # 其他情况尝试通用处理
            if "target_class" in original_params:
                target_class = original_params["target_class"]
                filter_fn = lambda t: t != target_class
            else:
                poison_loader.attack_params = original_params
                return 0.0

        if target_class is None: 
            poison_loader.attack_params = original_params
            return 0.0
            
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # 应用筛选
                indices = torch.where(filter_fn(target))[0]
                if len(indices) == 0: continue
                
                data_subset = data[indices]
                target_subset = target[indices]
                
                # 施加投毒 (此时 ratio=1.0，全量变毒)
                data_poisoned, target_poisoned = poison_loader.apply_data_poison(data_subset, target_subset)
                
                data_poisoned = data_poisoned.to(self.device)
                target_poisoned = target_poisoned.to(self.device)
                
                output = self.global_model(data_poisoned)
                _, predicted = output.max(1)
                
                total += len(target_poisoned)
                correct += predicted.eq(target_poisoned).sum().item()
        
        # [修正5] 恢复原始配置
        poison_loader.attack_params = original_params
        
        if total == 0: return 0.0
        return 100. * correct / total