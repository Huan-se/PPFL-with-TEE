# entity/Client.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from _utils_.poison_loader import PoisonLoader
from _utils_.tee_adapter import TEEAdapter 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Client:
    def __init__(self, client_id, dataloader, model_class, poison_loader=None, verbose=False, log_interval=100):
        self.client_id = client_id
        self.dataloader = dataloader
        self.model_class = model_class
        self.poison_loader = poison_loader or PoisonLoader()
        
        self.model = None
        self.optimizer = None
        
        # [修改] 替换 SuperBitLSH 为 TEEAdapter
        self.tee_adapter = TEEAdapter() 
        self.tee_adapter.initialize_enclave() # 启动 SGX
        
        # 保存用于 TEE 计算的旧参数 (w_old)
        self.w_old_flat = None 
        self.layer_indices = None
        
        # 保存投影种子 (从服务器接收)
        self.proj_seed = 42 
        
        self.verbose = verbose
        self.log_interval = log_interval

    def receive_model_and_proj(self, model_params, proj_seed=42):
        """
        接收全局模型和投影种子
        """
        if self.model is None:
            self.model = self.model_class().to(DEVICE)
        
        # 1. 加载参数到模型 (作为下一轮训练的起点)
        self.model.load_state_dict(model_params)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        
        # 2. [关键] 保存 w_old (训练前的参数状态)
        # 我们将其展平并转为 numpy，准备传给 TEE
        self.w_old_flat = self._flatten_model_params(self.model)
        
        # 3. 保存种子
        self.proj_seed = int(proj_seed)
        
        if self.layer_indices is None:
            self._calculate_layer_indices()

    def _flatten_model_params(self, model):
        """辅助：将模型参数展平为 1D float32 numpy 数组"""
        # 注意：必须严格按照 parameters() 的顺序
        all_params = []
        for param in model.parameters():
            all_params.append(param.data.view(-1).cpu().numpy())
        return np.concatenate(all_params).astype(np.float32)

    def _calculate_layer_indices(self):
        """辅助：计算每层参数的 (start, length)"""
        self.layer_indices = {}
        current_idx = 0
        for name, param in self.model.named_parameters():
            length = param.numel()
            self.layer_indices[name] = (current_idx, length)
            current_idx += length

    def local_train(self):
        """本地训练 (REE GPU 加速)"""
        if self.verbose:
            print(f"  > [Client {self.client_id}] Start Local Training...")

        # 注意：这里我们不再需要在 Python 侧计算 grad_flat，因为 TEE 会自己算
        # 但为了兼容 poison_loader 的攻击逻辑，我们保留它
        trained_params, grad_flat = self.poison_loader.execute_attack(
            self.model, 
            self.dataloader, 
            self.model_class, 
            DEVICE, 
            self.optimizer,
            verbose=self.verbose,
            uid=self.client_id,
            log_interval=self.log_interval
        )
        return grad_flat

    def generate_gradient_projection(self, target_layers=None):
        """
        [修改] 调用 TEE 进行梯度计算和投影
        """
        if self.w_old_flat is None:
            raise ValueError("Initial model state not set! Call receive_model_and_proj first.")
        
        # 1. 准备 w_new (训练后的参数)
        w_new_flat = self._flatten_model_params(self.model)
        
        projections = {}
        
        # 2. 计算区间 (Ranges)
        # TEE 需要知道我们要处理哪些部分：全量还是分层
        
        # A. 全量投影 (Full)
        # Range: [0, total_len]
        # 注意：这里我们不再在 Python 里算，而是直接让 TEE 算
        proj_full = self.tee_adapter.secure_project(
            seed=self.proj_seed,
            w_new_flat=w_new_flat,
            w_old_flat=self.w_old_flat,
            ranges=[[0, w_new_flat.size]], # 全量区间
            output_dim=1024
        )
        projections['full'] = proj_full # 这里可能还要加上 poison_loader 的攻击逻辑，暂时忽略
        
        # B. 指定层投影 (Layers)
        if target_layers:
            projections['layers'] = {}
            for layer_name in target_layers:
                if layer_name in self.layer_indices:
                    start, length = self.layer_indices[layer_name]
                    
                    # 调用 TEE 对单层进行投影
                    proj_layer = self.tee_adapter.secure_project(
                        seed=self.proj_seed,
                        w_new_flat=w_new_flat,
                        w_old_flat=self.w_old_flat,
                        ranges=[[start, length]], # 单层区间
                        output_dim=1024
                    )
                    projections['layers'][layer_name] = proj_layer
        
        return projections

    def prepare_upload_weighted_params(self, weight):
        """计算加权参数 (REE)"""
        current_params = self.model.state_dict()
        weighted_params = {}
        for key, param in current_params.items():
            if param.dtype in [torch.float32, torch.float64]:
                weighted_params[key] = param * weight
            else:
                weighted_params[key] = param 
        return weighted_params

    def __del__(self):
        # 析构时尝试关闭 Enclave
        if hasattr(self, 'tee_adapter'):
            self.tee_adapter.close()