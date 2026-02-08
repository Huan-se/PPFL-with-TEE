import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from _utils_.tee_adapter import get_tee_adapter_singleton

class Client(object):
    def __init__(self, client_id, train_loader, model_class, poison_loader, device_str='cuda', verbose=False):
        self.client_id = client_id
        self.train_loader = train_loader
        self.model_class = model_class 
        self.poison_loader = poison_loader 
        self.verbose = verbose
        
        if device_str == 'cuda' and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        self.model = model_class().to(self.device)
        
        # 默认参数 (会被 main.py 覆盖)
        self.learning_rate = 0.1 
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.local_epochs = 1
        
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
        self.model.load_state_dict(global_params)
        self.w_old_cache = self._flatten_params(self.model)

    def phase1_local_train(self, epochs=1):
        """[Phase 1] 本地训练 (集成 PoisonLoader)"""
        t_start = time.time()
        self.model.train()
        
        optimizer = optim.SGD(
            self.model.parameters(), 
            lr=self.learning_rate, 
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        
        run_epochs = epochs if epochs is not None else self.local_epochs

        # 梯度裁剪阈值
        MAX_GRAD_NORM = 10.0

        if self.poison_loader and self.poison_loader.attack_methods:
            self.poison_loader.attack_params['local_epochs'] = run_epochs
            
            # PoisonLoader 内部需要增加 clip_grad_norm 的逻辑，或者在这里只能信任它
            new_state_dict, _ = self.poison_loader.execute_attack(
                model=self.model,
                dataloader=self.train_loader,
                model_class=self.model_class,
                device=self.device,
                optimizer=optimizer,
                verbose=self.verbose,
                uid=self.client_id
            )
            self.model.load_state_dict(new_state_dict)
            
        else:
            criterion = nn.CrossEntropyLoss()
            for epoch in range(run_epochs):
                for batch_idx, (data, target) in enumerate(self.train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    output = self.model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=MAX_GRAD_NORM)
                    
                    optimizer.step()
        
        t_end = time.time()
        return t_end - t_start

    def phase2_tee_process(self, proj_seed):
        t_start = time.time()
        w_new_flat = self._flatten_params(self.model)
        
        if self.w_old_cache is None: 
            self.w_old_cache = np.zeros_like(w_new_flat)
        
        if np.isnan(w_new_flat).any(): 
            print(f"  [Warning] Client {self.client_id} has NaN weights!")
            w_new_flat = np.zeros_like(w_new_flat)

        output, ranges = self.tee_adapter.prepare_gradient(
            self.client_id, proj_seed, w_new_flat, self.w_old_cache
        )
        self.ranges = ranges
        
        t_end = time.time()
        data_size = len(self.train_loader.dataset) if hasattr(self.train_loader, 'dataset') else 1000
        return {'full': output}, data_size

    def tee_step1_encrypt(self, w, active_ids, seed_mask_root, seed_global_0):
        """Phase A: 仅生成加密梯度 (Cipher)"""
        w_new_flat = self._flatten_params(self.model)
        model_len = len(w_new_flat)
        if self.ranges is None: self.ranges = np.array([0, model_len], dtype=np.int32)
        
        # 调用 TEE 生成密文
        c_grad = self.tee_adapter.generate_masked_gradient_dynamic(
            seed_mask_root, seed_global_0, self.client_id, active_ids, 
            w, model_len
        )
        return c_grad

    def tee_step2_generate_shares(self, seed_sss, seed_mask_root, u1_ids, u2_ids):
        """Phase B: 根据服务器广播的在线列表 (U2) 生成 Shares"""
        # u1_ids: 本轮计划参与者 (active_ids)
        # u2_ids: 实际在线者 (Online Users)
        
        threshold = len(u1_ids) // 2 + 1
        
        # 调用 TEE 生成 Shares，传入真实的 U2，从而正确计算 Delta
        shares = self.tee_adapter.get_vector_shares_dynamic(
            seed_sss, seed_mask_root, 
            u1_ids, u2_ids, 
            self.client_id, threshold
        )
        return shares