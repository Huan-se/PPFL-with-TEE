import argparse
import yaml
import copy
import random
import torch
import numpy as np
import os
import sys
import time
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# 动态添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Entity.Server import Server
from Entity.Client import Client
from _utils_.dataloader import get_dataloader
from _utils_.poison_loader import PoisonLoader
from model.Cifar10Net import CIFAR10Net
from model.Lenet5 import LeNet5
from _utils_.save_config import save_result_with_config
from _utils_.tee_adapter import get_tee_adapter_singleton

# ==========================================
# 辅助函数
# ==========================================
def flatten_params(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1).cpu().numpy())
    return np.concatenate(params).astype(np.float32)

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# ==========================================
# 辅助类：投毒数据加载器包装器
# ==========================================
class PoisonedDataLoaderWrapper:
    def __init__(self, dataloader, poison_loader):
        self.dataloader = dataloader
        self.poison_loader = poison_loader
        self.dataset = dataloader.dataset

    def __iter__(self):
        for data, target in self.dataloader:
            data_p, target_p = self.poison_loader.apply_data_poison(data, target)
            yield data_p, target_p

    def __len__(self):
        return len(self.dataloader)

# ==========================================
# 并行任务包装器 (Wrapper Functions)
# ==========================================

def task_phase1_train(client, epochs):
    """Phase 1: 纯训练任务"""
    try:
        # 调用 Client 内部的 phase1_local_train
        train_time = client.phase1_local_train(epochs)
        return client.client_id, train_time, None
    except Exception as e:
        import traceback
        traceback.print_exc()
        return client.client_id, 0, str(e)

def task_phase2_tee(client, proj_seed):
    """Phase 2: 纯 TEE 处理任务"""
    try:
        # 调用 Client 内部的 phase2_tee_process
        feature_dict, data_size = client.phase2_tee_process(proj_seed)
        
        # 将 numpy 转换为 tensor 供 Server 检测模块使用 (CPU端)
        feature_dict_cpu = {'full': torch.from_numpy(feature_dict['full']), 'layers': {}}
        # 如果有分层特征，也进行转换
        if 'layers' in feature_dict and feature_dict['layers']:
            for k, v in feature_dict['layers'].items():
                feature_dict_cpu['layers'][k] = torch.from_numpy(v)
                
        return client.client_id, feature_dict_cpu, data_size, None
    except Exception as e:
        import traceback
        traceback.print_exc()
        return client.client_id, None, 0, str(e)

# ==========================================
# 主流程
# ==========================================
def run_single_mode(config, mode_name, mode_config):
    print(f"\n=================================================================")
    print(f"  Configuration Summary | Mode: {mode_name}")
    print(f"=================================================================")
    
    exp_conf = config['experiment']
    fed_conf = config['federated']
    data_conf = config['data']
    atk_conf = config['attack']
    
    seed = exp_conf['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    
    # 检测设备并设置策略
    device_str = exp_conf['device']
    if torch.cuda.is_available() and 'cuda' in device_str:
        print("  [System] CUDA GPU Detected. Optimized for Single-GPU Training.")
        device_str = 'cuda'
    else:
        print("  [System] Using CPU.")
        device_str = 'cpu'
    
    use_multiprocessing = exp_conf.get('use_multiprocessing', False)
    worker_count = exp_conf.get('thread_count', 4)
    
    print(f"  [Init] Loading dataset {data_conf['dataset']}...")
    # 使用真实的 get_dataloader
    train_loaders, test_loader = get_dataloader(
        data_conf['dataset'], 
        fed_conf['total_clients'], 
        fed_conf['batch_size'], 
        data_conf['if_noniid'], 
        data_conf['alpha']
    )

    if data_conf['model'] == 'cifar10': ModelClass = CIFAR10Net
    elif data_conf['model'] == 'mnist': ModelClass = LeNet5
    else: raise ValueError(f"Unknown model: {data_conf['model']}")
        
    # --- Attack Init ---
    poison_loader = None
    malicious_clients_list = []
    
    if atk_conf.get('active_attacks'):
        attack_methods = atk_conf['active_attacks']
        if isinstance(attack_methods, str): attack_methods = [attack_methods]
        
        print(f"  [Attack] Initializing PoisonLoader: {attack_methods}...")
        poison_loader = PoisonLoader(attack_methods, atk_conf)
        
        p_ratio = atk_conf.get('poison_ratio', 0.0)
        num_malicious = int(fed_conf['total_clients'] * p_ratio)
        if num_malicious > 0:
            malicious_clients_list = sorted(random.sample(range(fed_conf['total_clients']), num_malicious))
        
        poison_loader.malicious_set = set(malicious_clients_list)
        
        def is_poisoned_client(uid):
            return uid in poison_loader.malicious_set
        
        def get_poisoned_dataloader(uid, origin_loader):
            return PoisonedDataLoaderWrapper(origin_loader, poison_loader)
            
        poison_loader.is_poisoned_client = is_poisoned_client
        poison_loader.get_poisoned_dataloader = get_poisoned_dataloader
        
        print(f"  [Attack] Malicious Clients ({len(malicious_clients_list)}): {malicious_clients_list}")

    detection_method = mode_name if "detection" in mode_name else "none"
    if mode_name == 'pure_training': detection_method = "none"
    
    log_name = f"{mode_name}_{data_conf['dataset']}_{data_conf['model']}_{detection_method}_{atk_conf['active_attacks'] or 'NoAttack'}_p{atk_conf['poison_ratio']:.2f}_{'NonIID' if data_conf['if_noniid'] else 'IID'}"
    log_path = os.path.join(current_dir, "results", f"{log_name}_detection_log.csv")
    
    # --- Init Server ---
    server = Server(
        model_class=ModelClass,
        test_dataloader=test_loader,
        device_str=device_str,
        detection_method=detection_method, 
        defense_config=mode_config,
        seed=seed,
        verbose=True,
        log_file_path=log_path,
        malicious_clients=malicious_clients_list
    )
    
    # --- Init Clients ---
    clients = []
    for i in range(fed_conf['total_clients']):
        # 传递 device_str 确保 Client 使用正确的设备
        c = Client(i, train_loaders[i], ModelClass, poison_loader, device_str=device_str, verbose=(i==0))
        clients.append(c)

    # --- [关键] 预初始化 TEE ---
    print("  [System] Pre-initializing TEE Enclave (Global Singleton)...")
    adapter = get_tee_adapter_singleton()
    adapter.initialize_enclave()
    
    current_w_old_flat = flatten_params(server.global_model)
    total_rounds = fed_conf['comm_rounds']
    local_epochs = fed_conf['local_epochs']
    
    acc_history = []
    asr_history = []
    
    # === Main Loop ===
    for r in range(1, total_rounds + 1):
        print(f"\n>>> Round {r}/{total_rounds} Start...")
        round_start_time = time.time()
        
        # Step 1: Client Selection
        active_ids = random.sample(range(fed_conf['total_clients']), fed_conf['active_clients'])
        active_ids.sort()
        
        # Step 2: Global Model Distribution (Serial, Fast)
        global_params, _ = server.get_global_params_and_proj()
        global_params_cpu = {k: v.cpu() for k, v in global_params.items()}
        current_proj_seed = int(seed + r)
        
        # 客户端接收参数 (同时缓存 w_old)
        for cid in range(fed_conf['total_clients']):
             # 这里可以只给 active 的发，但为了保持状态一致性，通常全发
             clients[cid].receive_model(global_params)

        # -------------------------------------------------------------
        # Phase 1: Parallel Training
        # -------------------------------------------------------------
        # 策略：CUDA -> 串行 (workers=1); CPU -> 并行
        train_workers = 1 if device_str == "cuda" else min(5, worker_count)
        print(f"  [Phase 1] Starting Local Training (Device: {device_str}, Workers: {train_workers})...")
        t_p1_start = time.time()

        if use_multiprocessing:
            with ThreadPoolExecutor(max_workers=train_workers) as executor:
                futures = {
                    executor.submit(task_phase1_train, clients[cid], local_epochs): cid 
                    for cid in active_ids
                }
                for future in as_completed(futures):
                    cid, t_cost, err = future.result()
                    if err: print(f"  [Error] Client {cid} Train: {err}")
        else:
            # Fallback to serial
            for cid in active_ids:
                task_phase1_train(clients[cid], local_epochs)
        
        t_p1_end = time.time()
        print(f"  >> Training Phase Finished in {t_p1_end - t_p1_start:.2f}s")

        # -------------------------------------------------------------
        # Phase 2: Parallel TEE Processing
        # -------------------------------------------------------------
        # 策略：CPU Bound & GIL Released -> 全速并行
        tee_workers = worker_count 
        print(f"  [Phase 2] Starting TEE Processing (Workers: {tee_workers})...")
        t_p2_start = time.time()
        
        client_features_dict_list = {}
        client_data_sizes = {}
        
        if use_multiprocessing:
            with ThreadPoolExecutor(max_workers=tee_workers) as executor:
                futures = {
                    executor.submit(task_phase2_tee, clients[cid], current_proj_seed): cid 
                    for cid in active_ids
                }
                for future in as_completed(futures):
                    cid, feats, dsize, err = future.result()
                    if feats:
                        client_features_dict_list[cid] = feats
                        client_data_sizes[cid] = dsize
                    else:
                        print(f"  [Error] Client {cid} TEE: {err}")
        else:
            for cid in active_ids:
                _, feats, dsize, err = task_phase2_tee(clients[cid], current_proj_seed)
                if feats:
                    client_features_dict_list[cid] = feats
                    client_data_sizes[cid] = dsize

        t_p2_end = time.time()
        print(f"  >> TEE Phase Finished in {t_p2_end - t_p2_start:.2f}s")

        # --- Phase 3: Defense & Aggregation ---
        valid_ids = [cid for cid in active_ids if cid in client_features_dict_list]
        feature_list = [client_features_dict_list[cid] for cid in valid_ids]
        size_list = [client_data_sizes[cid] for cid in valid_ids]
        
        # 计算权重 (防御)
        weights_map = server.calculate_weights(valid_ids, feature_list, size_list, current_round=r)
        
        accepted_ids = [cid for cid, w in weights_map.items() if w > 1e-6]
        accepted_ids.sort()
        
        if not accepted_ids: 
            print("  [Warning] No accepted clients. Skipping aggregation.")
            continue

        # 安全聚合 (Server 内部串行调用 upload，但因为 Phase 2 做了准备，所以很快)
        server.secure_aggregation(clients, accepted_ids, round_num=r)

        current_w_old_flat = flatten_params(server.global_model)

        # 评估
        acc, loss = server.evaluate()
        acc_history.append(acc)
        
        round_end_time = time.time()
        print(f"  [Round {r}] Global Acc: {acc:.2f}%, Loss: {loss:.4f}")
        print(f"  Time Breakdown: Train={t_p1_end-t_p1_start:.1f}s, TEE={t_p2_end-t_p2_start:.1f}s, Agg={round_end_time-t_p2_end:.1f}s")
        
        if atk_conf.get('active_attacks'):
            attack_params = atk_conf.get('attack_params', {})
            asr = server.evaluate_asr(test_loader, atk_conf['active_attacks'], attack_params)
            asr_history.append(asr)
            print(f"  [Round {r}] ASR: {asr:.2f}%")
        else:
            asr_history.append(0.0)

        save_result_with_config(
            os.path.join(current_dir, "results"),
            mode_name,
            data_conf['model'],
            data_conf['dataset'],
            detection_method,
            config,
            acc_history,
            asr_history
        )

    print(f"\n[Done] Mode {mode_name} Finished.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../config/config.yaml')
    args = parser.parse_args()
    config = load_config(args.config)
    
    modes_from_config = config['experiment'].get('modes', 'pure_training')
    if isinstance(modes_from_config, str):
        mode_list = [modes_from_config]
    else:
        mode_list = modes_from_config

    for m in mode_list:
        run_config = {}
        if m == 'poison_with_detection':
            run_config = config.get('defense', {})
        elif m == 'poison_no_detection':
            run_config = {}
        elif m == 'pure_training':
            run_config = {}
        run_single_mode(config, m, run_config)

if __name__ == '__main__':
    main()