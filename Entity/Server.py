import torch
import torch.nn as nn
import copy
import numpy as np
import os
import csv
from collections import defaultdict

# 导入原有的防御和工具模块
from _utils_.LSH_proj_extra import SuperBitLSH
from Defence.score import ScoreCalculator
from Defence.kickout import KickoutManager
from Defence.layers_proj_detect import Layers_Proj_Detector

# [新增] 导入 Adapter 用于恢复掩码
from _utils_.tee_adapter import TEEAdapter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# [新增] 定义大素数 (必须与 C++ Enclave 保持一致)
MOD = 9223372036854775783
SCALE = 10000.0

class Server(object):
    def __init__(self, model_class, test_dataloader, device_str, detection_method="none", defense_config=None, seed=42, verbose=False, log_file_path=None, malicious_clients=None):
        """
        服务端初始化
        :param model_class: 模型类 (如 Cifar10Net)，非实例
        :param test_dataloader: 测试集加载器
        :param device_str: 设备字符串 'cuda' or 'cpu'
        :param detection_method: 防御方法名称
        :param defense_config: 防御参数配置字典
        :param seed: 随机种子
        :param verbose: 是否打印详细日志
        :param log_file_path: 结果日志路径
        :param malicious_clients: 恶意客户端ID列表 (用于日志标记)
        """
        self.device = torch.device(device_str)
        self.global_model = model_class().to(self.device)
        self.test_dataloader = test_dataloader
        
        # 定义损失函数
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        # [新增] TEE Adapter 初始化
        self.tee_adapter = TEEAdapter()

        # [保留] 防御与检测模块初始化
        self.superbit_lsh = SuperBitLSH(seed=seed)
        self.projection_matrix_path = None
        self.detection_method = detection_method
        self.verbose = verbose
        self.log_file_path = log_file_path
        
        self.malicious_clients = set(malicious_clients) if malicious_clients else set()
        self.defense_config = defense_config or {}
        
        # 状态追踪
        self.suspect_counters = {} 
        self.global_update_direction = None 
        self.detection_history = defaultdict(lambda: {'suspect_cnt': 0, 'kicked_cnt': 0, 'events': []})
        
        # 初始化具体防御器
        det_params = self.defense_config.get('params', {})
        self.mesas_detector = Layers_Proj_Detector(config=det_params)
        
        self.score_calculator = ScoreCalculator() if "score" in detection_method else None
        self.kickout_manager = KickoutManager() if "kickout" in detection_method else None
        self.current_round_weights = {}

        # [新增] 密钥管理上下文 (模拟 KeyServer 分发)
        # 在实际部署中，这些种子应通过 RA-TLS 安全协商
        self.seed_mask_root = 0x12345678 
        self.seed_global_0 = 0x87654321  
        self.seed_sss = 0x11223344

        # 初始化日志文件
        should_log = any(k in self.detection_method for k in ["mesas", "projected", "layers_proj"])
        if self.log_file_path and should_log:
            self._init_log_file()

    def _init_log_file(self):
        """初始化详细日志 CSV 文件"""
        log_dir = os.path.dirname(self.log_file_path)
        if log_dir: os.makedirs(log_dir, exist_ok=True)
        
        headers = ["Round", "Client_ID", "Type", "Score", "Status"]
        target_layers = self.defense_config.get('target_layers', [])
        scopes = ['full'] + [f'layer_{name}' for name in target_layers]
        metrics = ['l2', 'var', 'dist']
        
        # 动态生成表头
        for scope in scopes:
            for metric in metrics:
                base = f"{scope}_{metric}"
                headers.extend([base, f"{base}_threshold"])
        
        try:
            with open(self.log_file_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
        except Exception as e:
            print(f"  [Warning] Log init failed: {e}")

    def generate_projection_matrix(self, input_dim, output_dim, matrix_file_path=None):
        """[Phase 1] 生成并保存投影矩阵 (用于 LSH)"""
        if matrix_file_path is None:
            matrix_file_path = f"proj/projection_matrix_{input_dim}x{output_dim}.pt"
        self.projection_matrix_path = self.superbit_lsh.generate_projection_matrix(
            input_dim, output_dim, device='cpu', matrix_file_path=matrix_file_path
        )

    def get_global_params_and_proj(self):
        """[Phase 1] 下发全局参数和投影矩阵路径"""
        return copy.deepcopy(self.global_model.state_dict()), self.projection_matrix_path

    def calculate_weights(self, client_id_list, client_features_dict_list, client_data_sizes, current_round=0):
        """
        [Phase 3] 执行防御策略，计算每个客户端的聚合权重
        """
        # 构建 {client_id: features} 字典
        client_projections = {cid: feat for cid, feat in zip(client_id_list, client_features_dict_list)}
        
        self._update_global_direction_feature(client_projections)
        weights = {}

        # 1. 使用 MESAS / Layers_Proj 检测
        if any(k in self.detection_method for k in ["mesas", "projected", "layers_proj"]):
            if self.verbose:
                print(f"  [Server] Executing {self.detection_method} detection (Round {current_round})...")

            raw_weights, logs, global_stats = self.mesas_detector.detect(
                client_projections, 
                self.global_update_direction, 
                self.suspect_counters,
                verbose=self.verbose 
            )
            
            # 记录日志
            if self.log_file_path:
                self._write_detection_log(current_round, logs, raw_weights, global_stats)

            # 统计被踢出/怀疑次数
            for cid in sorted(logs.keys()):
                status = logs[cid].get('status', 'NORMAL')
                if "SUSPECT" in status:
                    self.detection_history[cid]['suspect_cnt'] += 1
                if "KICKED" in status:
                    self.detection_history[cid]['kicked_cnt'] += 1

            # 归一化权重
            total_score = sum(raw_weights.values())
            if total_score > 0:
                weights = {cid: s / total_score for cid, s in raw_weights.items()}
            else:
                weights = {cid: 0.0 for cid in raw_weights}
            
            self._update_global_direction_feature(client_projections)
            
        else:
            # 2. 回退到旧的检测方法 (Score / Kickout / None)
            full_features = [f['full'] for f in client_features_dict_list]
            weights = self._fallback_old_detection(client_id_list, full_features, client_data_sizes)

        # 保存本轮权重供 Secure Aggregation 使用
        self.current_round_weights = weights
        return weights

    # =========================================================================
    # [新增] Phase 4 & 5: 安全聚合核心逻辑
    # =========================================================================

    def secure_aggregation(self, client_objects, active_ids, round_num):
        """
        执行完整的安全聚合流程：收集密文 -> 掉线恢复 -> 消除掩码 -> 更新模型
        :param client_objects: 客户端对象列表 (必须包含所有 active_ids 中的客户端)
        :param active_ids: Phase 3 选出的参与者列表 (U1)
        :param round_num: 当前轮次
        """
        if self.verbose:
            print(f"\n[Server] Starting Secure Aggregation (Round {round_num})...")

        # 1. 准备权重
        weights_map = self.current_round_weights
        
        # 2. [Phase 4] 收集加密梯度 (Collection)
        encrypted_grads = {}
        online_clients = [] # U2 列表 (成功上传梯度的客户端)
        
        total_params_len = sum(p.numel() for p in self.global_model.parameters())

        # 模拟网络通信：向 active_ids 请求数据
        for client in client_objects:
            if client.client_id not in active_ids: continue
            
            w = weights_map.get(client.client_id, 0.0)
            if w <= 1e-9: 
                # 权重为0也被视为未参与聚合，或者需要特殊处理
                # 这里我们假设权重为0的客户端不参与聚合计算
                continue

            try:
                # Client TEE: 生成加密梯度
                # 必须传入 seed_global_0 (本轮通过某种方式更新或固定)
                c_grad = client.tee_step2_upload(
                    w, active_ids, self.seed_mask_root, self.seed_global_0
                )
                encrypted_grads[client.client_id] = c_grad
                online_clients.append(client.client_id)
            except Exception as e:
                print(f"  [Warning] Client {client.client_id} failed to upload: {e}")

        if not online_clients:
            print("  [Error] No clients uploaded gradients. Aggregation aborted.")
            return

        # 3. [Phase 5] 掉线恢复 (Secret Recovery)
        u2_ids = sorted(online_clients)
        if self.verbose:
            print(f"  [Server] Online clients (U2): {u2_ids} (Total: {len(u2_ids)})")

        # 设定恢复阈值 (Threshold)
        # 注意: 阈值必须 <= len(u2_ids)，因为只有 u2_ids 能提供分片
        # 且必须 > 1 (至少2个点才能确定一次函数，除非是常数函数)
        threshold = int(len(u2_ids) * 0.6) + 1 
        if threshold < 2: threshold = 1
        if threshold > len(u2_ids): threshold = len(u2_ids)

        # 收集秘密分片
        shares_collected = [] # List[Dict]
        
        for client in client_objects:
            if client.client_id in u2_ids:
                try:
                    # Client TEE: 生成针对 U2 的分片
                    share_vec = client.tee_step3_get_shares(
                        self.seed_sss, self.seed_mask_root, 
                        active_ids, u2_ids, threshold
                    )
                    shares_collected.append({
                        'x': client.client_id + 1, # x 坐标 (ID + 1)
                        'v': share_vec             # y 向量
                    })
                except Exception as e:
                    print(f"  [Warning] Failed to get share from Client {client.client_id}: {e}")

        if len(shares_collected) < threshold:
            print(f"  [Error] Insufficient shares collected ({len(shares_collected)} < {threshold}).")
            return
        
        # 4. 秘密重构 (Lagrange Interpolation)
        # 恢复出的向量 S = [Delta, Alpha, Beta_u2_1, Beta_u2_2, ...]
        secret_vector = self._reconstruct_secrets(shares_collected, threshold)
        if secret_vector is None:
            print("  [Error] Failed to reconstruct secrets.")
            return

        # 解析秘密向量
        delta = secret_vector[0]      # 掉线补偿系数
        alpha_seed = secret_vector[1] # 全局掩码种子参数
        beta_seeds = secret_vector[2:] # 在线用户的自掩码种子
        
        if len(beta_seeds) != len(u2_ids):
            print(f"  [Error] Reconstructed beta seeds count mismatch! Expected {len(u2_ids)}, got {len(beta_seeds)}")
            # 这是一个严重错误，通常意味着 threshold 恢复出了错误数据或者逻辑不对齐
            return

        # 5. 消除掩码与聚合 (Unmasking)
        if self.verbose:
            print("  [Server] Unmasking and aggregating...")

        # (A) 聚合密文: Sum(C_i)
        # 使用 Object 类型进行大数运算，防止 int64 溢出
        agg_cipher_obj = np.zeros(total_params_len, dtype=object)
        
        for cid in u2_ids:
            # 将 int64 numpy 数组转为 object 数组进行累加
            agg_cipher_obj += encrypted_grads[cid].astype(object)
            # 为了节省内存，每加几次可以取一次模，这里简化为最后取模
            # 如果内存不足，需要分块处理
        
        agg_cipher_obj %= MOD
        
        # (B) 生成掩码向量 (使用 TEEAdapter 调用 C++ 保持一致性)
        
        # B-1. 全局掩码向量 M
        # M = PRG(seed_global_0 + alpha)
        # 注意: 这里的加法和掩码必须与 C++ (seed_global_0 + seed_alpha) & 0x7FFFFFFF 逻辑一致
        final_seed_M = (self.seed_global_0 + alpha_seed) & 0x7FFFFFFF
        vec_M = self.tee_adapter.generate_noise_from_seed(final_seed_M, total_params_len).astype(object)
        
        # B-2. 自掩码总和 Sum(B_i)
        vec_B_sum = np.zeros(total_params_len, dtype=object)
        for b_seed in beta_seeds:
            # 对每个在线用户生成其 B_i
            vec_B = self.tee_adapter.generate_noise_from_seed(b_seed, total_params_len).astype(object)
            vec_B_sum += vec_B
        
        vec_B_sum %= MOD

        # (C) 消除掩码
        # 公式: Agg' = Sum(C) - (1 - Delta) * M - Sum(B)
        
        # 计算全局掩码的保留系数: coeff = 1 - Delta
        # 注意模减法: (1 - Delta) % MOD
        coeff_M = (1 - delta) % MOD
        
        # 计算 Term_M = coeff * M
        term_M = (coeff_M * vec_M) % MOD
        
        # 最终消除: Result = Agg_Cipher - Term_M - Term_B
        # 转换为加法: Result = Agg_Cipher - Term_M - Term_B + 2*MOD ...
        result_int = (agg_cipher_obj - term_M - vec_B_sum) % MOD
        # 调整到正数范围
        result_int = (result_int + MOD) % MOD
        
        # 6. 反定点化与更新 (Inverse Quantization)
        # 规则: 数值 > MOD/2 则视为负数
        threshold_neg = MOD // 2
        
        # 将 object 转回 float (需要处理负数映射)
        # 1. 识别负数
        mask_neg = result_int > threshold_neg
        
        # 2. 转换
        # 这里创建一个临时的 object 数组处理减法，因为 int64 可能会在减 MOD 时溢出(虽然是负数)
        # 但 result_int 已经是正数且 < MOD，减去 MOD 后的结果绝对值 < MOD，可以用 int64 存
        # 不过为了安全，我们先减，再转 float
        
        temp_arr = np.array(result_int) # object array
        temp_arr[mask_neg] -= MOD # 还原回负数 (Python 大整数支持负数)
        
        # 3. 除以 SCALE 并转为 float32
        result_float = temp_arr.astype(np.float32) / SCALE
        
        # 7. 应用更新到模型
        self._apply_global_update(result_float)
        
        # 更新 seed_global_0 为下一轮做准备 (可选)
        self.seed_global_0 = (self.seed_global_0 + 1) & 0x7FFFFFFF

        if self.verbose:
            print("  [Server] Secure aggregation completed successfully.")

    def _reconstruct_secrets(self, shares, threshold):
        """
        拉格朗日插值恢复秘密向量
        :param shares: List[Dict{'x', 'v'}]
        :param threshold: 最小恢复阈值
        :return: 恢复出的 secrets 向量 (numpy array int64)
        """
        if len(shares) < threshold: return None
        
        # 只取前 threshold 个分片即可
        selected_shares = shares[:threshold]
        
        # 获取向量长度
        vec_len = len(selected_shares[0]['v'])
        reconstructed = np.zeros(vec_len, dtype=np.int64)
        
        x_s = [s['x'] for s in selected_shares]
        
        # 对向量的每一位进行插值恢复
        # 性能注意: 这里有两层循环 (vec_len * threshold)，如果 secrets 很长会慢
        # 但 secrets 长度仅为 2 + client_num，非常短，所以没问题
        for i in range(vec_len):
            y_s = [s['v'][i] for s in selected_shares]
            
            # 计算 f(0) 即为秘密值
            secret_val = self._lagrange_interpolate_zero(x_s, y_s)
            reconstructed[i] = secret_val
            
        return reconstructed

    def _lagrange_interpolate_zero(self, x_s, y_s):
        """
        计算 L(0) 在 mod MOD 下的值
        Formula: L(0) = Sum( y_j * Product( (0 - x_m)/(x_j - x_m) ) )
        """
        total = 0
        k = len(x_s)
        for j in range(k):
            # 计算基函数 lj(0)
            numerator = 1
            denominator = 1
            for m in range(k):
                if m == j: continue
                
                # numerator *= (0 - xm) -> -xm
                numerator = (numerator * (0 - x_s[m])) % MOD
                
                # denominator *= (xj - xm)
                diff = x_s[j] - x_s[m]
                denominator = (denominator * diff) % MOD
            
            # 计算分母的模逆
            # Python 3.8+ pow(a, -1, m) 支持模逆
            inv_den = pow(int(denominator), -1, int(MOD))
            
            lj_0 = (numerator * inv_den) % MOD
            
            term = (y_s[j] * lj_0) % MOD
            total = (total + term) % MOD
            
        return (total + MOD) % MOD

    def _apply_global_update(self, update_flat):
        """
        将扁平的 float 更新量应用到 global_model
        逻辑: w_global = w_global + Aggregated_Gradient
        """
        idx = 0
        
        # 确保 update_flat 在正确的设备上
        # update_flat 是 numpy float32
        
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                numel = param.numel()
                grad_segment = update_flat[idx : idx+numel]
                
                # 转为 Tensor
                grad_tensor = torch.from_numpy(grad_segment).view(param.shape).to(self.device)
                
                # 更新参数
                # 这里的逻辑取决于 update_flat 是 "新的权重" 还是 "梯度"
                # 根据 C++ 逻辑: G = (w_new - w_old) * k
                # 所以 Agg(G) 是加权后的梯度和
                # 因此 w_new_global = w_old_global + Agg(G)
                param.data.add_(grad_tensor)
                
                idx += numel

    # =========================================================================
    # 原有辅助方法
    # =========================================================================

    def _update_global_direction_feature(self, client_projections):
        """更新全局动量方向 (保留原逻辑占位符)"""
        if not client_projections: return
        pass

    def _fallback_old_detection(self, ids, features, sizes):
        """兼容旧版检测逻辑 (纯训练、Score、Kickout)"""
        # 无防御：按数据量加权
        if not self.score_calculator and not self.kickout_manager:
            total_size = sum(sizes)
            if total_size > 0:
                return {cid: size / total_size for cid, size in zip(ids, sizes)}
            else:
                return {cid: 1.0 / len(ids) for cid in ids}
        
        # 计算分数
        client_scores = {}
        for i, cid in enumerate(ids):
            feat = features[i]
            if isinstance(feat, torch.Tensor): feat = feat.cpu().numpy()
            client_scores[cid] = self.score_calculator.calculate_scores(cid, feat, sizes[i])
            
        weights = {}
        if self.kickout_manager:
            weights = self.kickout_manager.determine_weights(client_scores)
        else:
            # 仅评分不踢人
            raw_scores = {cid: s['final_score'] for cid, s in client_scores.items()}
            total_s = sum(raw_scores.values())
            if total_s > 0:
                weights = {cid: s / total_s for cid, s in raw_scores.items()}
            else:
                weights = {cid: 1.0 / len(ids) for cid in ids}
        return weights

    def _write_detection_log(self, round_num, logs, raw_weights, stats):
        """将全量+分层的详细数据写入 CSV"""
        target_layers = self.defense_config.get('target_layers', [])
        scopes = ['full'] + [f'layer_{name}' for name in target_layers]
        metrics_list = ['l2', 'var', 'dist']
        
        try:
            with open(self.log_file_path, 'a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                for cid in sorted(logs.keys()):
                    metrics = logs[cid]
                    client_type = "Malicious" if cid in self.malicious_clients else "Benign"
                    
                    row = [
                        round_num, cid, client_type, 
                        f"{raw_weights.get(cid, 0):.2f}", 
                        metrics.get('status', 'UNKNOWN')
                    ]
                    
                    for scope in scopes:
                        for metric in metrics_list:
                            val = metrics.get(f"{scope}_{metric}", 0)
                            thresh = stats.get(f"{scope}_{metric}_threshold", 0)
                            row.extend([f"{val:.4f}", f"{thresh:.4f}"])
                    writer.writerow(row)
        except Exception:
            pass

    def evaluate(self):
        """在测试集上评估全局模型"""
        self.global_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_dataloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.global_model(inputs)
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100. * correct / total
        avg_loss = test_loss / len(self.test_dataloader)
        return acc, avg_loss

    def evaluate_asr(self, test_loader, attack_type, attack_params):
        """评估攻击成功率 (ASR) - 保留接口"""
        self.global_model.eval()
        correct_attack = 0
        total_attack = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                if attack_type == "label_flip":
                    source_class = attack_params.get("source_class", 5)
                    target_class = attack_params.get("target_class", 7)
                    mask = (target == source_class)
                    if mask.sum() == 0: continue
                    data_source = data[mask]
                    outputs = self.global_model(data_source)
                    _, predicted = torch.max(outputs.data, 1)
                    correct_attack += (predicted == target_class).sum().item()
                    total_attack += mask.sum().item()
                    
                elif attack_type == "backdoor":
                    target_class = attack_params.get("backdoor_target", 0)
                    trigger_size = attack_params.get("trigger_size", 3)
                    data_poisoned = data.clone()
                    if data_poisoned.dim() == 4:
                         data_poisoned[:, :, -trigger_size:, -trigger_size:] = data_poisoned.max()
                    outputs = self.global_model(data_poisoned)
                    _, predicted = torch.max(outputs.data, 1)
                    correct_attack += (predicted == target_class).sum().item()
                    total_attack += data.size(0)
                    
        if total_attack == 0: return 0.0
        return 100 * correct_attack / total_attack

    def update_global_model_with_state_dict(self, new_state_dict):
        """
        [Phase 5] 使用安全聚合后的参数更新全局模型
        (注意：如果是 Secure Aggregation 模式，该函数可能不再被调用，而是直接在 secure_aggregation 中更新)
        """
        self.global_model.load_state_dict(new_state_dict)