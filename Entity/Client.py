import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from _utils_.tee_adapter import TEEAdapter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Client(object):
    def __init__(self, client_id, train_loader, model_class, poison_loader, verbose=False):
        self.client_id = client_id
        self.train_loader = train_loader
        self.model = model_class().to(DEVICE)
        self.poison_loader = poison_loader 
        self.verbose = verbose
        
        # [参数设置]
        self.learning_rate = 0.1 
        self.momentum = 0.9
        
        self.tee_adapter = TEEAdapter()
        self.ranges = None 
        
        # [新增] 用于缓存 Phase 1 之前的旧参数 (w_t)，供 Phase 2 计算 (w_t+1 - w_t)
        self.w_old_cache = None

    def _get_poisoned_dataloader(self):
        return self.train_loader

    def _flatten_params(self, model):
        params = []
        for param in model.parameters():
            params.append(param.data.view(-1).cpu().numpy())
        return np.concatenate(params).astype(np.float32)

    def receive_model(self, global_params):
        """[Phase 0] 接收全局模型并缓存"""
        self.model.load_state_dict(global_params)
        # 缓存当前参数作为 w_old，用于后续计算梯度
        self.w_old_cache = self._flatten_params(self.model)

    def phase1_local_train(self, epochs=1):
        """[Phase 1] 并行训练阶段 (GPU/CPU 计算密集型)"""
        t_start = time.time()
        
        self.model.train()
        
        # [可选] 验证同步状态 (Pre-train Acc)
        # if self.verbose:
        #     correct_pre = 0
        #     total_pre = 0
        #     with torch.no_grad():
        #         # 只抽样 1 个 batch 快速验证，避免拖慢整体速度
        #         for i, (data, target) in enumerate(self._get_poisoned_dataloader()):
        #             if i > 0: break
        #             data, target = data.to(DEVICE), target.to(DEVICE)
        #             output = self.model(data)
        #             _, predicted = output.max(1)
        #             total_pre += target.size(0)
        #             correct_pre += predicted.eq(target).sum().item()
        #     if total_pre > 0:
        #         print(f"  [Client {self.client_id}] Pre-train Acc (Sample): {100.*correct_pre/total_pre:.2f}%")

        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(self._get_poisoned_dataloader()):
                data, target = data.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        t_end = time.time()
        if self.verbose:
            print(f"  [Time] Client {self.client_id} Training: {t_end - t_start:.4f}s")
        return t_end - t_start

    def phase2_tee_process(self, proj_seed):
        """[Phase 2] TEE 数据处理阶段 (CPU 逻辑密集型)"""
        t_start = time.time()
        
        if self.tee_adapter is None:
            raise RuntimeError("TEE Adapter not initialized")

        # 1. 获取训练后的新参数
        w_new_flat = self._flatten_params(self.model)
        
        # 2. 获取旧参数
        if self.w_old_cache is None:
            self.w_old_cache = np.zeros_like(w_new_flat)
            
        # [安全检查]
        if np.isnan(w_new_flat).any() or np.isinf(w_new_flat).any():
            print(f"  [Warning] Client {self.client_id} model has NaN/Inf! Resetting.")
            w_new_flat = np.zeros_like(w_new_flat)

        # 3. 调用 Enclave 计算投影 (耗时操作)
        ret_tuple = self.tee_adapter.prepare_gradient(
            self.client_id, proj_seed, w_new_flat, self.w_old_cache
        )
        
        # 处理返回值 (兼容性处理)
        if len(ret_tuple) == 4:
            output, out_len, ranges, r_len = ret_tuple
        elif len(ret_tuple) == 5:
            _, output, out_len, ranges, r_len = ret_tuple
        else:
            output, out_len, ranges, r_len = ret_tuple[-4:]

        self.ranges = ranges
        
        t_end = time.time()
        # 返回投影特征供 Server 检测，以及 ranges 供后续使用
        return {'full': output}, len(w_new_flat)

    def tee_step2_upload(self, w, active_ids, seed_mask_root, seed_global_0):
        """[Phase 4] TEE 加密上传 (Server 串行调用，快速)"""
        w_new_flat = self._flatten_params(self.model)
        model_len = len(w_new_flat)
        if self.ranges is None: self.ranges = np.array([0, model_len], dtype=np.int32)
        
        c_grad = self.tee_adapter.generate_masked_gradient_dynamic(
            seed_mask_root, seed_global_0, self.client_id, active_ids, 
            w, w_new_flat, self.ranges, model_len
        )
        return c_grad

    def tee_step3_get_shares(self, seed_sss, seed_mask_root, active_ids, u2_ids, threshold):
        """[Phase 5] 掉线恢复"""
        return self.tee_adapter.get_vector_shares_dynamic(
            seed_sss, seed_mask_root, active_ids, u2_ids, 
            self.client_id, threshold
        )