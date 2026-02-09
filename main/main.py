import sys
import os
import argparse
import yaml
import time
import numpy as np
import torch
import copy
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# 路径修复
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Entity.Client import Client
from Entity.Server import Server
from _utils_.tee_adapter import get_tee_adapter_singleton
# [新增] 引入 ServerAdapter 以便设置日志开关
from _utils_.server_adapter import ServerAdapter 
from _utils_.dataloader import load_and_split_dataset
from _utils_.poison_loader import PoisonLoader
from _utils_.save_config import save_result_with_config, check_result_exists, get_result_filename
from model.Cifar10Net import CIFAR10Net
from model.Lenet5 import LeNet5

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# 任务函数
def task_phase1_train(client, epochs):
    try:
        return client.client_id, client.phase1_local_train(epochs), None
    except Exception as e:
        return client.client_id, 0, str(e)

def task_phase2_tee(client, proj_seed):
    try:
        feature_dict, data_size = client.phase2_tee_process(proj_seed)
        feature_dict_cpu = {'full': torch.from_numpy(feature_dict['full']), 'layers': {}}
        if 'layers' in feature_dict and feature_dict['layers']:
            for k, v in feature_dict['layers'].items():
                feature_dict_cpu['layers'][k] = torch.from_numpy(v)
        return client.client_id, feature_dict_cpu, data_size, None
    except Exception as e:
        return client.client_id, None, 0, str(e)

def flatten_params(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1).cpu().numpy())
    return np.concatenate(params).astype(np.float32)

def run_single_mode(full_config, mode_name, current_mode_config):
    # 获取 verbose 开关 (默认为 False)
    exp_conf = full_config['experiment']
    verbose_flag = exp_conf.get('verbose', False)

    # [新增] 定义受控打印函数
    def log(*args, **kwargs):
        if verbose_flag:
            print(*args, **kwargs)

    print(f"\n=================================================================")
    print(f"  Configuration Summary | Mode: {mode_name}")
    print(f"  Verbose Logging: {'ON' if verbose_flag else 'OFF'}")
    print(f"=================================================================")
    
    fed_conf = full_config['federated']
    data_conf = full_config['data']
    atk_conf = full_config['attack']
    defense_conf = full_config['defense']
    
    seed = exp_conf['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False
    
    device_str = exp_conf.get('device', 'auto')
    if device_str == 'auto':
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  [System] Using Device: {device_str}")
    
    use_multiprocessing = exp_conf.get('use_multiprocessing', False)
    worker_count = exp_conf.get('thread_count', 4)
    log_interval = exp_conf.get('log_interval', 100)
    
    save_dir = os.path.join(current_dir, "results")
    
    # 检查结果是否存在
    detection_method = current_mode_config.get('defense_method', 'none')
    exists, _ = check_result_exists(
        save_dir, mode_name, data_conf['model'], data_conf['dataset'], 
        detection_method, current_mode_config
    )
    if exists:
        print(f"[Skip] Mode {mode_name} already exists.")
        return

    # Init Data
    print(f"  [Init] Loading dataset {data_conf['dataset']}...")
    all_client_dataloaders, test_loader = load_and_split_dataset(
        dataset_name=data_conf['dataset'],
        num_clients=fed_conf['total_clients'],
        batch_size=fed_conf['batch_size'],
        if_noniid=data_conf['if_noniid'],
        alpha=data_conf['alpha'],
        data_dir="./data"
    )
    
    if data_conf['model'] == 'cifar10': ModelClass = CIFAR10Net
    elif data_conf['model'] == 'lenet5': ModelClass = LeNet5
    else: raise ValueError(f"Unknown model: {data_conf['model']}")

    # Init Attack
    poison_client_ids = []
    current_poison_ratio = current_mode_config.get('poison_ratio', 0.0)
    if current_poison_ratio > 0:
        poison_client_ids = random.sample(range(fed_conf['total_clients']), int(fed_conf['total_clients'] * current_poison_ratio))
        poison_client_ids.sort()
        print(f"  [Attack] Malicious Clients ({len(poison_client_ids)}): {poison_client_ids}")
    else:
        print(f"  [Attack] Malicious Clients: None")

    # 准备日志路径
    log_file_path = None
    if any(k in detection_method for k in ["mesas", "projected", "layers_proj"]):
        log_filename = get_result_filename(mode_name, data_conf['model'], data_conf['dataset'], detection_method, current_mode_config).replace('.npz', '_detection_log.csv')
        log_file_path = os.path.join(save_dir, log_filename)
    
    # Init Server
    server = Server(
        model_class=ModelClass,
        test_dataloader=test_loader,
        device_str=device_str,
        detection_method=detection_method, 
        defense_config=defense_conf,
        seed=seed,
        verbose=verbose_flag,               # [修改] 跟随 config 开关
        log_file_path=log_file_path,
        malicious_clients=poison_client_ids
    )
    
    # Init Clients
    clients = []
    active_attacks = atk_conf.get('active_attacks', [])
    attack_params_dict = atk_conf.get('params', {})
    attack_idx = 0
    primary_attack_type = active_attacks[0] if active_attacks else None
    
    server_poison_loader = None
    if primary_attack_type:
        server_poison_loader = PoisonLoader([primary_attack_type], attack_params_dict.get(primary_attack_type, {}))

    for cid in range(fed_conf['total_clients']):
        client_poison_loader = None
        if cid in poison_client_ids and active_attacks:
            a_type = active_attacks[attack_idx % len(active_attacks)]
            attack_idx += 1
            a_params = attack_params_dict.get(a_type, {})
            client_poison_loader = PoisonLoader([a_type], a_params)
        
        # [修改] 只有当 verbose_flag=True 且是第一个客户端时，才允许 Client 内部打印
        c_verbose = (verbose_flag and cid == 0)
        c = Client(cid, all_client_dataloaders[cid], ModelClass, client_poison_loader, device_str=device_str, verbose=c_verbose, log_interval=log_interval)
        c.learning_rate = fed_conf.get('lr', 0.01)
        c.local_epochs = fed_conf.get('local_epochs', 1)
        clients.append(c)

    # Pre-init TEE & Configure Logging
    print("  [System] Pre-initializing TEE Enclave (Global Singleton)...")
    
    # 1. 获取 TEE Adapter 实例并初始化
    tee_adapter = get_tee_adapter_singleton()
    tee_adapter.initialize_enclave()
    
    # 2. [关键] 设置 TEE (Enclave + Bridge) 的日志开关
    # 增加 try-catch 防止 C++ 库未更新导致 Crash
    try:
        if hasattr(tee_adapter, 'set_verbose'):
            tee_adapter.set_verbose(verbose_flag)
    except Exception as e:
        print(f"  [Warning] Failed to set TEE verbose: {e}")
    
    # 3. [关键] 设置 ServerCore (C++ 聚合层) 的日志开关
    try:
        temp_server_adapter = ServerAdapter()
        if hasattr(temp_server_adapter, 'set_verbose'):
            temp_server_adapter.set_verbose(verbose_flag)
    except Exception as e:
        print(f"  [Warning] Failed to set ServerCore verbose: {e}")

    total_rounds = fed_conf['comm_rounds']
    acc_history = []
    asr_history = []
    loss_history = []
    
    start_time = time.time()
    
    for r in range(1, total_rounds + 1):
        # 使用 log() 替代 print()，受 verbose_flag 控制
        log(f"\n>>> Round {r}/{total_rounds} Start...")
        round_start_time = time.time()
        
        active_ids = random.sample(range(fed_conf['total_clients']), fed_conf['active_clients'])
        active_ids.sort()
        
        global_params, _ = server.get_global_params_and_proj()
        current_proj_seed = int(seed + r)
        
        for c in clients: c.receive_model(global_params)

        # Phase 1
        train_workers = 1 if "cuda" in device_str else min(5, worker_count)
        log(f"  [Phase 1] Starting Local Training (Device: {device_str}, Workers: {train_workers})...")
        t_p1_start = time.time()
        if use_multiprocessing:
            with ThreadPoolExecutor(max_workers=train_workers) as executor:
                futures = {executor.submit(task_phase1_train, clients[cid], None): cid for cid in active_ids}
                for future in as_completed(futures):
                    cid, t_cost, err = future.result()
                    if err: print(f"  [Error] Client {cid} Train: {err}")
        else:
            for cid in active_ids: task_phase1_train(clients[cid], None)
        t_p1_end = time.time()
        log(f"  >> Training Phase Finished in {t_p1_end - t_p1_start:.2f}s")

        # Phase 2
        tee_workers = worker_count 
        log(f"  [Phase 2] Starting TEE Processing (Workers: {tee_workers})...")
        t_p2_start = time.time()
        client_features_dict_list = {}
        client_data_sizes = {}
        
        if use_multiprocessing:
            with ThreadPoolExecutor(max_workers=tee_workers) as executor:
                futures = {executor.submit(task_phase2_tee, clients[cid], current_proj_seed): cid for cid in active_ids}
                for future in as_completed(futures):
                    cid, feats, dsize, err = future.result()
                    if feats:
                        client_features_dict_list[cid] = feats
                        client_data_sizes[cid] = dsize
                    else: print(f"  [Error] Client {cid} TEE: {err}")
        else:
            for cid in active_ids:
                _, feats, dsize, err = task_phase2_tee(clients[cid], current_proj_seed)
                if feats:
                    client_features_dict_list[cid] = feats
                    client_data_sizes[cid] = dsize
        t_p2_end = time.time()
        log(f"  >> TEE Phase Finished in {t_p2_end - t_p2_start:.2f}s")

        # Phase 3
        valid_ids = [cid for cid in active_ids if cid in client_features_dict_list]
        feature_list = [client_features_dict_list[cid] for cid in valid_ids]
        size_list = [client_data_sizes[cid] for cid in valid_ids]
        
        weights_map = server.calculate_weights(valid_ids, feature_list, size_list, current_round=r)
        
        accepted_ids = [cid for cid, w in weights_map.items() if w > 1e-6]
        accepted_ids.sort()
        
        if not accepted_ids: 
            print("  [Warning] No accepted clients. Skipping aggregation.")
            continue

        server.secure_aggregation(clients, accepted_ids, round_num=r)

        # Eval
        acc, loss = server.evaluate()
        acc_history.append(acc)
        loss_history.append(loss)
        
        round_end_time = time.time()
        
        # [关键] 无论 verbose 如何，Round 结果始终打印
        print(f"  [Round {r}] Global Acc: {acc:.2f}%, Loss: {loss:.4f}")
        
        # if server_poison_loader and current_poison_ratio > 0:
        if server_poison_loader:
            asr = server.evaluate_asr(test_loader, server_poison_loader)
            asr_history.append(asr)
            print(f"  [Round {r}] ASR: {asr:.2f}%")
        else:
            asr_history.append(0.0)
            
        log(f"  Time Breakdown: Train={t_p1_end-t_p1_start:.1f}s, TEE={t_p2_end-t_p2_start:.1f}s, Agg={round_end_time-t_p2_end:.1f}s")

    print("\n[Saving] Saving final results...")
    save_result_with_config(
        save_dir,
        mode_name,
        data_conf['model'],
        data_conf['dataset'],
        detection_method,
        current_mode_config,
        acc_history,
        asr_history,
        loss_history=loss_history
    )
    print(f"[Done] Mode {mode_name} Finished.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../config/config.yaml')
    parser.add_argument('--mode', type=str, default=None, help='指定只运行某个模式(逗号分隔)')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    base_flat_config = {
        'total_clients': config['federated']['total_clients'],
        'batch_size': config['federated']['batch_size'],
        'comm_rounds': config['federated']['comm_rounds'],
        'if_noniid': config['data']['if_noniid'],
        'alpha': config['data']['alpha'],
        'attack_types': config['attack']['active_attacks'],
        'seed': config['experiment']['seed'],
        'model_type': config['data']['model'],
        'dataset_type': config['data']['dataset']
    }
    
    default_poison_ratio = config['attack']['poison_ratio']
    default_defense = config['defense']['method']

    all_modes = [
        {
            'name': 'pure_training', 
            'mode_config': {**base_flat_config, 'poison_ratio': 0.0, 'defense_method': 'none'}
        },
        {
            'name': 'poison_no_detection', 
            'mode_config': {**base_flat_config, 'poison_ratio': default_poison_ratio, 'defense_method': 'none'}
        },
        {
            'name': 'poison_with_detection', 
            'mode_config': {**base_flat_config, 'poison_ratio': default_poison_ratio, 'defense_method': default_defense}
        }
    ]
    
    target_modes_str = args.mode
    if target_modes_str is None: target_modes_str = config['experiment'].get('modes', 'all')
    
    if target_modes_str == 'all':
        modes_to_run = all_modes
    else:
        target_names = [m.strip() for m in target_modes_str.split(',')]
        modes_to_run = [m for m in all_modes if m['name'] in target_names]

    print(f"计划运行模式: {[m['name'] for m in modes_to_run]}")

    for mode in modes_to_run:
        run_single_mode(config, mode['name'], mode['mode_config'])

if __name__ == '__main__':
    main()