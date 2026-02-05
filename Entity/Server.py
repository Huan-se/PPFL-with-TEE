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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        # [修改] 实例化模型并移动到设备
        self.device = torch.device(device_str)
        self.global_model = model_class().to(self.device)
        self.test_dataloader = test_dataloader
        
        # 定义损失函数
        self.criterion = nn.CrossEntropyLoss().to(self.device)

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
        这是原有代码的核心防御逻辑。
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
                # 如果所有人都被踢了，本轮作废 (权重全0)
                weights = {cid: 0.0 for cid in raw_weights}
            
            self._update_global_direction_feature(client_projections)
            
        else:
            # 2. 回退到旧的检测方法 (Score / Kickout / None)
            full_features = [f['full'] for f in client_features_dict_list]
            weights = self._fallback_old_detection(client_id_list, full_features, client_data_sizes)

        self.current_round_weights = weights
        return weights

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
        """
        self.global_model.load_state_dict(new_state_dict)