import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

# 尝试导入 TEE Adapter
try:
    from _utils_.tee_adapter import TEEAdapter
except ImportError:
    TEEAdapter = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Client(object):
    def __init__(self, client_id, dataloader, model_class, poison_loader=None, verbose=False):
        self.client_id = client_id
        self.dataloader = dataloader
        self.model_class = model_class # 保存类引用以便重新实例化
        self.model = model_class().to(DEVICE)
        self.poison_loader = poison_loader
        self.verbose = verbose
        
        self.learning_rate = 0.01 
        self.momentum = 0.9
        
        # TEE Adapter (由 main.py 初始化或在此处初始化)
        self.tee_adapter = None
        if TEEAdapter:
            self.tee_adapter = TEEAdapter()
            
    def receive_model(self, global_state_dict):
        """接收并加载全局模型参数"""
        self.model.load_state_dict(global_state_dict)

    def _get_poisoned_dataloader(self):
        """检查并获取投毒数据加载器"""
        # 适配 Monkey Patched 的方法
        if self.poison_loader and hasattr(self.poison_loader, 'is_poisoned_client'):
            if self.poison_loader.is_poisoned_client(self.client_id):
                if self.verbose:
                    # 为了防止多线程刷屏，这里可以注释掉，或者保留
                    # print(f"  [Client {self.client_id}] Loading poisoned data...")
                    pass
                # 适配 Monkey Patched 的 get_poisoned_dataloader
                return self.poison_loader.get_poisoned_dataloader(self.client_id, self.dataloader)
        return self.dataloader

    def local_train(self, epochs=1):
        """[Phase 2 - Part A] 本地训练"""
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        criterion = nn.CrossEntropyLoss()
        
        train_loader = self._get_poisoned_dataloader()
        
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(DEVICE), target.to(DEVICE)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                # 统计
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

            # [新增] 训练日志输出
            if self.verbose:
                avg_loss = total_loss / len(train_loader)
                acc = 100. * correct / total
                print(f"  [Client {self.client_id}] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

    def _flatten_params(self, model):
        params = []
        for param in model.parameters():
            params.append(param.data.view(-1).cpu().numpy())
        return np.concatenate(params).astype(np.float32)

    def tee_step1_prepare(self, w_old_flat, proj_seed, target_layers_config=None):
        """[Phase 2 - Part B] TEE 准备阶段"""
        if self.tee_adapter is None:
            raise RuntimeError("TEE Adapter not initialized")

        w_new_flat = self._flatten_params(self.model)
        
        if w_old_flat is None:
            w_old_flat = np.zeros_like(w_new_flat)
            
        # 调用 TEE prepare
        projection = self.tee_adapter.prepare_gradient(
            self.client_id,
            proj_seed,
            w_new_flat,
            w_old_flat
        )
        
        feature_dict = {
            'full': projection,
            'layers': {} 
        }
        
        return feature_dict, len(self.dataloader.dataset)

    def tee_step2_upload(self, weight, active_ids, seed_mask_root, seed_global_0):
        """[Phase 4] TEE 上传阶段 - 生成加密梯度"""
        if self.tee_adapter is None:
            raise RuntimeError("TEE Adapter not initialized")
            
        # 计算模型参数总长度
        w_len = sum(p.numel() for p in self.model.parameters())

        # 调用 TEE generate_masked_gradient
        encrypted_grad = self.tee_adapter.generate_masked_gradient(
            seed_mask=seed_mask_root,
            seed_g0=seed_global_0,
            cid=self.client_id,
            active_ids=active_ids, 
            k_weight=weight,        
            model_len=w_len
        )
        
        return encrypted_grad

    def tee_step3_get_shares(self, seed_sss, seed_mask_root, u1_ids, u2_ids, threshold):
        """[Phase 5] 掉线恢复 - 生成秘密分片"""
        if self.tee_adapter is None:
            raise RuntimeError("TEE Adapter not initialized")
            
        my_share_vector = self.tee_adapter.get_vector_shares(
            seed_sss=seed_sss,
            seed_mask=seed_mask_root,
            u1_ids=u1_ids, 
            u2_ids=u2_ids,
            my_cid=self.client_id,
            threshold=threshold
        )
        return my_share_vector