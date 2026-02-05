import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

# 尝试导入 TEE Adapter，如果环境不支持则在运行时报错
try:
    from _utils_.tee_adapter import TEEAdapter
except ImportError:
    TEEAdapter = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Client(object):
    def __init__(self, client_id, dataloader, model_class, poison_loader=None, verbose=False):
        self.client_id = client_id
        self.dataloader = dataloader
        self.model = model_class().to(DEVICE)
        self.poison_loader = poison_loader # 保留投毒模块
        self.verbose = verbose
        
        # 优化器配置 (可根据 config 调整，这里保持默认 SGD)
        self.learning_rate = 0.01 
        self.momentum = 0.9
        
        # 初始化 TEE 适配器
        self.tee_adapter = None
        if TEEAdapter:
            self.tee_adapter = TEEAdapter()
            
    def receive_model(self, global_state_dict):
        """接收并加载全局模型参数"""
        self.model.load_state_dict(global_state_dict)

    def _get_poisoned_dataloader(self):
        """检查是否有针对当前客户端的投毒任务"""
        if self.poison_loader and self.poison_loader.is_poisoned_client(self.client_id):
            if self.verbose:
                print(f"  [Client {self.client_id}] Loading poisoned data...")
            return self.poison_loader.get_poisoned_dataloader(self.client_id, self.dataloader)
        return self.dataloader

    def local_train(self, epochs=1):
        """
        [Phase 2 - Part A] 本地训练 (包含攻击逻辑)
        """
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        criterion = nn.CrossEntropyLoss()
        
        # 获取数据加载器 (可能是被投毒的)
        train_loader = self._get_poisoned_dataloader()
        
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(DEVICE), target.to(DEVICE)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
        # 训练结束，self.model 已更新为 w_new
        # 注意：这里不返回梯度，梯度计算移交给 TEE

    def _flatten_params(self, model):
        """辅助函数：将模型参数展平为 numpy float32 数组"""
        params = []
        for param in model.parameters():
            params.append(param.data.view(-1).cpu().numpy())
        return np.concatenate(params).astype(np.float32)

    def tee_step1_prepare(self, w_old_flat, proj_seed, target_layers_config=None):
        """
        [Phase 2 - Part B] TEE 准备阶段:
        1. 计算梯度 (w_new - w_old)
        2. 生成投影 (Projection)
        3. 锁定梯度状态 (Stateful Lock)
        
        Args:
            w_old_flat: 上一轮的全局模型 (numpy array)
            proj_seed: 投影矩阵种子
            target_layers_config: (未使用，保留接口兼容性)
        Returns:
            feature_dict: 包含投影信息的字典 {'full': proj_vec}
        """
        if self.tee_adapter is None:
            raise RuntimeError("TEE Adapter not initialized")

        # 1. 获取 w_new
        w_new_flat = self._flatten_params(self.model)
        
        # 确保 w_old 存在
        if w_old_flat is None:
            w_old_flat = np.zeros_like(w_new_flat)
            
        # 2. 调用 TEE (ecall_prepare_gradient)
        # TEE 内部会计算 diff, 存入 Buffer, 并返回投影
        # 注意：这里我们假设 TEE 返回的是 full projection。
        # 如果需要分层投影 (Layer-wise)，TEE 接口需要相应调整。
        # 为简化对接，目前假设 TEE 返回整体投影。
        
        projection = self.tee_adapter.prepare_gradient(
            self.client_id,
            proj_seed,
            w_new_flat,
            w_old_flat
        )
        
        # 构造符合 Server 防御接口的格式
        # Server 期望: {'full': np.array(...), 'layers': {...}}
        # 目前 TEE 只实现了 full projection
        feature_dict = {
            'full': projection,
            'layers': {} # 如果需要分层防御，需扩展 TEE 接口
        }
        
        return feature_dict, len(self.dataloader.dataset)

    def tee_step2_upload(self, weight, seed_mask_root, seed_global_0, n_ratio):
        """
        [Phase 4] TEE 上传阶段:
        1. 传入权重 k_i
        2. TEE 使用锁定的梯度计算 C_i = k_i * G + Masks
        3. 销毁梯度状态
        
        Returns:
            encrypted_gradient: int64 numpy array
        """
        if self.tee_adapter is None:
            raise RuntimeError("TEE Adapter not initialized")
            
        # 调用 TEE (ecall_generate_masked_gradient_dynamic)
        # 注意：不再传入 w_new/w_old
        encrypted_grad = self.tee_adapter.generate_masked_gradient(
            seed_mask_root,
            seed_global_0,
            self.client_id,
            weight, # k_weight
            n_ratio # n_ratio
        )
        
        return encrypted_grad

    def tee_step3_get_shares(self, seed_sss, seed_mask_root, target_id, threshold, total_clients):
        """
        [Phase 5] 掉线恢复协助:
        生成针对 target_id 的密钥分片
        """
        if self.tee_adapter is None:
            raise RuntimeError("TEE Adapter not initialized")
            
        # TEE 返回所有分片，取自己的那一份
        all_shares = self.tee_adapter.get_vector_shares(
            seed_sss,
            seed_mask_root,
            target_id,
            threshold,
            total_clients
        )
        # 返回本客户端持有的那份分片 (index = self.client_id)
        return all_shares[self.client_id]