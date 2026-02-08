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
        self.model_class = model_class # [新增] 保存类引用，供 PoisonLoader 使用
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
        epoch_grad_norms = []
        # [修改] 优先使用 PoisonLoader 接管训练
        # 检查是否为恶意客户端且有攻击方法
        if self.poison_loader and self.poison_loader.attack_methods:
            # PoisonLoader.execute_attack 会处理循环、投毒和反向传播
            # 注意：execute_attack 内部可能会 reset 模型参数，所以需要小心处理
            # 这里的 execute_attack 设计是返回 trained_params, grad_flat
            
            # 为了适配 PoisonLoader 的接口，我们需要临时传递参数
            # 注意：PoisonLoader 内部可能会重新创建 optimizer，或者我们传进去
            
            # 更新 PoisonLoader 的 local_epochs 参数 (如果需要覆盖)
            self.poison_loader.attack_params['local_epochs'] = run_epochs
            
            new_state_dict, _ = self.poison_loader.execute_attack(
                model=self.model,
                dataloader=self.train_loader,
                model_class=self.model_class,
                device=self.device,
                optimizer=optimizer,
                verbose=self.verbose,
                uid=self.client_id
            )
            # 加载被投毒/训练后的参数
            self.model.load_state_dict(new_state_dict)
            
        else:
            criterion = nn.CrossEntropyLoss()
            for epoch in range(run_epochs):
                batch_norms = []
                for batch_idx, (data, target) in enumerate(self.train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    output = self.model(data)
                    loss = criterion(output, target)
                    
                    # [检查] Loss 是否为 NaN
                    if torch.isnan(loss):
                        print(f"  [Error] Client {self.client_id} Loss is NaN at Epoch {epoch} Batch {batch_idx}!")
                        continue

                    loss.backward()
                    
                    # [关键调试] 计算梯度范数 (在裁剪之前)
                    total_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    batch_norms.append(total_norm)

                    # [防御] 梯度裁剪 (Clip)
                    # 建议将阈值设为 10.0 或更小
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    
                    optimizer.step()
                
                # 记录本 Epoch 平均梯度范数
                avg_norm = sum(batch_norms) / len(batch_norms) if batch_norms else 0
                epoch_grad_norms.append(avg_norm)

        t_end = time.time()
        
        # [调试输出] 如果梯度异常大，打印出来
        if epoch_grad_norms:
            max_norm = max(epoch_grad_norms)
            if max_norm > 20.0 or self.verbose: # 阈值可调
                print(f"  [Client {self.client_id}] Max Grad Norm: {max_norm:.4f}, local train err")

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
        # 返回数据量供加权使用
        data_size = len(self.train_loader.dataset) if hasattr(self.train_loader, 'dataset') else 1000
        return {'full': output}, data_size

    def tee_step2_upload(self, w, active_ids, seed_mask_root, seed_global_0):
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