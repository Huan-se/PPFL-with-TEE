import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from _utils_.tee_adapter import get_tee_adapter_singleton # [修改]

class Client(object):
    def __init__(self, client_id, train_loader, model_class, poison_loader, device_str='cuda', verbose=False):
        self.client_id = client_id
        self.train_loader = train_loader
        self.poison_loader = poison_loader 
        self.verbose = verbose
        if device_str == 'cuda' and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model = model_class().to(self.device)
        
        # 默认超参数 (会被 main.py 覆盖)
        self.learning_rate = 0.1 
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.local_epochs = 1
        
        # [修改] 使用单例 TEE 实例
        self.tee_adapter = get_tee_adapter_singleton()
        self.ranges = None 
        self.w_old_cache = None

    def _get_poisoned_dataloader(self):
        return self.train_loader

    def _flatten_params(self, model):
        params = []
        for param in model.parameters():
            params.append(param.data.view(-1).cpu().numpy())
        return np.concatenate(params).astype(np.float32)

    def receive_model(self, global_params):
        """[Phase 0]"""
        self.model.load_state_dict(global_params)
        # [关键] 缓存旧参数用于 Phase 2 差分
        self.w_old_cache = self._flatten_params(self.model)

    def phase1_local_train(self, epochs=1):
        """[Phase 1] 本地训练"""
        t_start = time.time()
        self.model.train()
        
        # 使用传入的 epochs
        e = epochs if epochs is not None else self.local_epochs
        
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(e):
            for batch_idx, (data, target) in enumerate(self._get_poisoned_dataloader()):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        return time.time() - t_start

    def phase2_tee_process(self, proj_seed):
        """[Phase 2] TEE 处理"""
        t_start = time.time()
        w_new_flat = self._flatten_params(self.model)
        
        # 防御性编程：如果是第一轮或缓存丢失，用零填充
        if self.w_old_cache is None: 
            self.w_old_cache = np.zeros_like(w_new_flat)
        
        if np.isnan(w_new_flat).any(): 
            print(f"  [Warning] Client {self.client_id} has NaN weights!")
            w_new_flat = np.zeros_like(w_new_flat)

        # 调用 TEE: 计算梯度并投影
        # [修复] 显式解包 tuple: (output_proj, ranges)
        output, ranges = self.tee_adapter.prepare_gradient(
            self.client_id, proj_seed, w_new_flat, self.w_old_cache
        )
        self.ranges = ranges
        
        # 返回 output 给 Server 做检测，返回 len 给 Phase 4
        return {'full': output}, len(w_new_flat)

    def tee_step2_upload(self, w, active_ids, seed_mask_root, seed_global_0):
        """[Phase 4] 加密上传"""
        # 注意：这里不需要计算梯度了，直接传长度即可，TEE 内部 map 已经存了梯度
        # 但为了接口兼容，我们还是传入 w_new_flat (TEEAdapter 会忽略它或者仅用它取长度)
        w_new_flat = self._flatten_params(self.model)
        model_len = len(w_new_flat)
        
        if self.ranges is None: self.ranges = np.array([0, model_len], dtype=np.int32)
        
        c_grad = self.tee_adapter.generate_masked_gradient_dynamic(
            seed_mask_root, seed_global_0, self.client_id, active_ids, 
            w, w_new_flat, self.ranges, model_len
        )
        return c_grad

    def tee_step3_get_shares(self, seed_sss, seed_mask_root, active_ids, u2_ids, threshold):
        return self.tee_adapter.get_vector_shares_dynamic(
            seed_sss, seed_mask_root, active_ids, u2_ids, self.client_id, threshold
        )