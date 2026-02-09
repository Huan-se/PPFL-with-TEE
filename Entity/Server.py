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
        
        # [修改] 扩展表头：包含 Type 和详细指标列
        headers = ["Round", "Client_ID", "Type", "Score", "Status"]
        
        # 动态生成指标列 (Full + Layers) * (L2, Var, Dist) * (Value, Median, Threshold)
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
            if self.verbose: print(f"  [Server] Executing {self.detection_method} detection (Round {current_round})...")
            
            # 执行检测，获取 global_stats
            raw_weights, logs, global_stats = self.mesas_detector.detect(
                client_projections, self.global_update_direction, self.suspect_counters, verbose=self.verbose
            )
            
            # [修改] 写入详细日志
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
                    
                    # 标记客户端类型 (Benign / Malicious)
                    c_type = "Malicious" if cid in self.malicious_clients else "Benign"
                    
                    row = [round_num, cid, c_type, f"{score:.4f}", status]
                    
                    # 填充详细指标
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
        # [计时] 总开始时间
        t_start_total = time.time()
        
        print(f"\n[Server] >>> STARTING V5 SECURE AGGREGATION (ROUND {round_num}) [Strict Demo] <<<")
        weights_map = self.current_round_weights
        sorted_active_ids = sorted(active_ids) # U1: 计划参与者

        # ==========================================
        # Step 1: 第一次握手 - 收集密文，确认在线名单 (U2)
        # ==========================================
        t_s1_start = time.time()
        print(f"  [Step 1] Broadcasting request & Collecting Ciphers...")
        
        u2_cids = []
        cipher_map = {} 
        
        for client in client_objects:
            # 仅处理本轮活跃的客户端
            if client.client_id not in sorted_active_ids: continue
            
            # 权重检查 (忽略权重极小的客户端)
            w = weights_map.get(client.client_id, 0.0)
            if w <= 1e-9: continue
            
            try:
                # Client: Phase A - 仅上传密文
                # 该操作包括 TEE 内梯度加掩码
                c_grad = client.tee_step1_encrypt(w, sorted_active_ids, 
                                                self.seed_mask_root, self.seed_global_0)
                u2_cids.append(client.client_id)
                cipher_map[client.client_id] = c_grad
            except Exception as e:
                print(f"    [Drop] Client {client.client_id} failed to upload cipher: {e}")

        u2_ids = sorted(u2_cids) # U2: 实际在线者 (Commitment)
        
        t_s1_end = time.time()
        print(f"  [Server] Confirmed Online Users (U2): {len(u2_ids)}/{len(sorted_active_ids)}")
        
        # 阈值检查
        threshold = len(sorted_active_ids) // 2 + 1
        if len(u2_ids) < threshold:
            print(f"  [Abort] Not enough clients! Need {threshold}, got {len(u2_ids)}.")
            return

        # ==========================================
        # Step 2: 第二次握手 - 广播 U2，收集 Shares
        # ==========================================
        t_s2_start = time.time()
        print(f"  [Step 2] Broadcasting U2 & Collecting Shares...")
        
        shares_list = []
        final_ciphers = []

        # 严格遍历 U2 列表，确保顺序一致
        for cid in u2_ids:
            # 模拟网络发送：找到对应的 Client 对象
            client = next(c for c in client_objects if c.client_id == cid)
            
            try:
                # Client: Phase B - 根据确定的 U2 计算 Delta 并生成 Share
                shares = client.tee_step2_generate_shares(
                    self.seed_sss, self.seed_mask_root,
                    sorted_active_ids, # U1 (用于确定向量长度)
                    u2_ids             # U2 (用于计算 Delta 和插值)
                )
                shares_list.append(shares)
                final_ciphers.append(cipher_map[cid])
            except Exception as e:
                 print(f"    [Error] Client {cid} dropped during Share generation: {e}")
                 # 这里不进行处理，由下方的长度检查触发 Abort

        # [关键逻辑] 原子性检查
        # 如果有人在 Step 2 掉线，剩余人的 Delta 计算将基于错误的 U2 假设，因此必须终止。
        if len(shares_list) != len(u2_ids):
            print(f"  [Abort] Consistency Broken! U2 size={len(u2_ids)}, Shares received={len(shares_list)}.")
            print(f"          (A client dropped after U2 commitment, invalidating the math.)")
            return

        t_s2_end = time.time()

        # ==========================================
        # Step 3: C++ 核心聚合 (ServerCore)
        # ==========================================
        t_agg_start = time.time()
        print("  [Step 3] Executing ServerCore Aggregation...")
        
        try:
            # 调用 C++ 动态库进行插值恢复与去噪
            result_int = self.server_adapter.aggregate_and_unmask(
                self.seed_mask_root,
                self.seed_global_0,
                sorted_active_ids, # U1: 用于映射 Beta 种子
                u2_ids,            # U2: 用于拉格朗日插值坐标
                shares_list,       # 份额矩阵
                final_ciphers      # 密文矩阵
            )
            
            # 反量化与更新全局模型
            threshold_neg = MOD // 2
            result_float = result_int.astype(np.float64)
            # 处理负数 (模域 -> 实数域)
            mask_neg = result_int > threshold_neg
            result_float[mask_neg] -= float(MOD)
            # 除以缩放因子
            result_float /= SCALE
            
            # 应用梯度更新
            self._apply_global_update(result_float)
            
            # 更新全局种子 (防止重放)
            self.seed_global_0 = (self.seed_global_0 + 1) & 0x7FFFFFFF
            print("  [Success] Aggregation Completed.")
            
        except Exception as e:
            print(f"  [Critical Error] Aggregation crashed: {e}")
            import traceback
            traceback.print_exc()

        t_agg_end = time.time()
        
        # ==========================================
        # 性能统计输出
        # ==========================================
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

    # 保留未使用的辅助函数，保持代码结构一致
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