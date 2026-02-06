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

# ==========================================
# 全局常量
# ==========================================
MOD = 9223372036854775783
SCALE = 10000.0
SIMULATED_DROP_RATE = 0.0

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# ==========================================
# 辅助函数
# ==========================================
def flatten_params(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1).cpu().numpy())
    return np.concatenate(params).astype(np.float32)

def unflatten_params(flat_params, reference_model):
    updated_state_dict = {}
    current_idx = 0
    for name, param in reference_model.named_parameters():
        numel = param.numel()
        shape = param.shape
        flat_slice = flat_params[current_idx : current_idx + numel]
        tensor_slice = torch.from_numpy(flat_slice).view(shape).to(param.device)
        updated_state_dict[name] = tensor_slice
        current_idx += numel
    return updated_state_dict

def generate_noise_vector(seed, shape):
    rng = np.random.Generator(np.random.MT19937(seed))
    flat_len = np.prod(shape)
    raw_u32 = rng.integers(0, 4294967296, size=flat_len * 2, dtype=np.uint32)
    pairs = raw_u32.reshape(-1, 2)
    low = pairs[:, 0].astype(np.uint64)
    high = pairs[:, 1].astype(np.uint64)
    raw_u64 = (high << 32) | low
    return (raw_u64 % MOD).astype(np.int64).reshape(shape)

def lagrange_interpolate(shares, x=0):
    secret = 0
    for i, (xi, yi) in enumerate(shares):
        num = 1
        den = 1
        for j, (xj, yj) in enumerate(shares):
            if i == j: continue
            num = (num * (x - xj)) % MOD
            den = (den * (xi - xj)) % MOD
        if den == 0: continue
        inv = pow(int(den), int(MOD) - 2, int(MOD))
        term = (yi * num * inv) % MOD
        secret = (secret + term) % MOD
    return secret

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
# 并行任务 (Thread Safe)
# ==========================================

def task_tee_step1_train_prepare(client, global_params_cpu, w_old_flat, proj_seed, target_layers, device_str):
    try:
        device = torch.device(device_str)
        if client.model is None:
            client.model = client.model_class().to(device)
        else:
            if next(client.model.parameters()).device != device:
                 client.model = client.model.to(device)
        
        global_params_device = {k: v.to(device) for k, v in global_params_cpu.items()}
        if hasattr(client, 'receive_model'):
            client.receive_model(global_params_device)
        else:
            client.model.load_state_dict(global_params_device)
            
        client.local_train()
        
        feature_dict, data_size = client.tee_step1_prepare(w_old_flat, proj_seed, target_layers)
        
        feature_dict_cpu = {'full': torch.from_numpy(feature_dict['full']), 'layers': {}}
        if 'layers' in feature_dict:
            for k, v in feature_dict['layers'].items():
                feature_dict_cpu['layers'][k] = torch.from_numpy(v)
                
        return client.client_id, feature_dict_cpu, data_size, None
    except Exception as e:
        import traceback
        traceback.print_exc()
        return client.client_id, None, 0, str(e)

def task_tee_step2_upload(client, weight, seed_mask_root, seed_global_0, n_ratio):
    try:
        if random.random() < SIMULATED_DROP_RATE:
            return client.client_id, None, "Dropped"
        
        encrypted_grad = client.tee_step2_upload(weight, seed_mask_root, seed_global_0, n_ratio)
        return client.client_id, encrypted_grad, None
    except Exception as e:
        return client.client_id, None, str(e)

def task_tee_step3_shares(client, seed_sss, seed_mask_root, target_id, threshold, total_clients):
    try:
        # share 是字典 {'alpha': ..., 'beta': ...}
        share = client.tee_step3_get_shares(seed_sss, seed_mask_root, target_id, threshold, total_clients)
        # [Fix] 直接返回字典，不要使用 share.alpha
        return client.client_id, share
    except Exception as e:
        # [Fix] 打印错误详情，而不是静默失败
        print(f"  [Error] Share recovery failed for client {client.client_id}: {e}")
        return client.client_id, None

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
    
    device_str = exp_conf['device']
    
    use_multiprocessing = exp_conf.get('use_multiprocessing', False)
    worker_count = exp_conf.get('thread_count', 4)
    
    if use_multiprocessing:
        print(f"  [System] Parallel Mode Enabled: Using {worker_count} Threads.")
        ExecutorClass = ThreadPoolExecutor
    else:
        print("  [System] Serial Mode.")
        ExecutorClass = None 

    print(f"  [Init] Loading dataset {data_conf['dataset']}...")
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
        
    # Attack Init
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
    
    # Global Enclave Init
    from _utils_.tee_adapter import TEEAdapter
    try:
        dummy_adapter = TEEAdapter()
        dummy_adapter.initialize_enclave()
        print("  [TEE] Enclave successfully initialized (Global).")
    except Exception as e:
        print(f"  [TEE] Enclave init warning: {e}")

    clients = []
    for i in range(fed_conf['total_clients']):
        c = Client(i, train_loaders[i], ModelClass, poison_loader)
        c.tee_adapter = TEEAdapter() 
        c.tee_adapter.initialized = True 
        clients.append(c)
        
    current_w_old_flat = flatten_params(server.global_model)
    total_rounds = fed_conf['comm_rounds']
    target_layers_config = mode_config.get('target_layers', None)
    
    acc_history = []
    asr_history = []
    
    # === Main Loop ===
    for r in range(1, total_rounds + 1):
        print(f"\n>>> Round {r}/{total_rounds} Start...")
        start_time = time.time()
        
        active_ids = random.sample(range(fed_conf['total_clients']), fed_conf['active_clients'])
        active_ids.sort()
        
        global_params, _ = server.get_global_params_and_proj()
        global_params_cpu = {k: v.cpu() for k, v in global_params.items()}
        current_proj_seed = int(seed + r)
        
        # --- Phase 2: Train & Prepare ---
        client_features_dict_list = {}
        client_data_sizes = {}
        
        if ExecutorClass:
            with ExecutorClass(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(
                        task_tee_step1_train_prepare, 
                        clients[cid], global_params_cpu, current_w_old_flat, current_proj_seed, target_layers_config, device_str
                    ): cid for cid in active_ids
                }
                for future in as_completed(futures):
                    cid, feats, dsize, err = future.result()
                    if feats:
                        client_features_dict_list[cid] = feats
                        client_data_sizes[cid] = dsize
                    else:
                        print(f"  [Error] Client {cid}: {err}")
        else:
            for cid in active_ids:
                _, feats, dsize, err = task_tee_step1_train_prepare(
                    clients[cid], global_params_cpu, current_w_old_flat, current_proj_seed, target_layers_config, device_str
                )
                if feats:
                    client_features_dict_list[cid] = feats
                    client_data_sizes[cid] = dsize

        # --- Phase 3: Defense ---
        valid_ids = [cid for cid in active_ids if cid in client_features_dict_list]
        feature_list = [client_features_dict_list[cid] for cid in valid_ids]
        size_list = [client_data_sizes[cid] for cid in valid_ids]
        
        weights_map = server.calculate_weights(valid_ids, feature_list, size_list, current_round=r)
        accepted_ids = [cid for cid, w in weights_map.items() if w > 1e-6]
        
        kicked = len(valid_ids) - len(accepted_ids)
        if kicked > 0: print(f"  [Defence] Kicked {kicked} clients.")
        if not accepted_ids: continue

        # --- Phase 4: Secure Upload ---
        SEED_mask_root = random.randint(0, 2**30)
        SEED_sss = random.randint(0, 2**30)
        Seed_Global_0 = int(hash(str(r)) & 0x7FFFFFFF)
        
        uploaded_c_i = {}
        online_clients = []
        
        if ExecutorClass:
            with ExecutorClass(max_workers=worker_count) as executor:
                futures = {}
                for cid in accepted_ids:
                    w = weights_map[cid]
                    futures[executor.submit(
                        task_tee_step2_upload, clients[cid], w, SEED_mask_root, Seed_Global_0, w
                    )] = cid
                for f in as_completed(futures):
                    cid, c_i, err = f.result()
                    if c_i is not None:
                        uploaded_c_i[cid] = c_i
                        online_clients.append(cid)
                    else:
                        if err != "Dropped": print(f"  [Error] Upload {cid}: {err}")
        else:
            for cid in accepted_ids:
                w = weights_map[cid]
                _, c_i, err = task_tee_step2_upload(clients[cid], w, SEED_mask_root, Seed_Global_0, w)
                if c_i is not None:
                    uploaded_c_i[cid] = c_i
                    online_clients.append(cid)

        # --- Phase 5: Aggregation ---
        if not uploaded_c_i: continue
        
        sample_shape = next(iter(uploaded_c_i.values())).shape
        agg_accumulator = np.zeros(sample_shape, dtype=np.int64)
        for cid in online_clients:
            agg_accumulator = (agg_accumulator + uploaded_c_i[cid]) % MOD
            
        threshold = 3
        if len(online_clients) < threshold: continue
        helpers = online_clients[:threshold]
        
        def fetch_share(helper_id, target_id):
            _, share_struct = task_tee_step3_shares(clients[helper_id], SEED_sss, SEED_mask_root, target_id, threshold, fed_conf['total_clients'])
            if share_struct is None: return 0 # Fallback safety
            if target_id == 0: return share_struct['alpha']
            return share_struct['beta']

        pts_alpha = [(h+1, fetch_share(h, 0)) for h in helpers]
        rec_alpha = lagrange_interpolate(pts_alpha)
        
        rec_betas = {}
        for u in online_clients:
            pts_beta = [(h+1, fetch_share(h, u)) for h in helpers]
            rec_betas[u] = lagrange_interpolate(pts_beta)

        real_global_seed = (Seed_Global_0 + int(rec_alpha)) & 0x7FFFFFFF
        vec_g = generate_noise_vector(real_global_seed, sample_shape)
        
        sum_weights = sum([weights_map[cid] for cid in online_clients])
        term_g = (vec_g * int(sum_weights * SCALE)) // int(SCALE)
        agg_accumulator = (agg_accumulator - term_g) % MOD
        
        for cid in online_clients:
            beta_seed = int(rec_betas[cid]) & 0x7FFFFFFF
            vec_b = generate_noise_vector(beta_seed, sample_shape)
            agg_accumulator = (agg_accumulator - vec_b) % MOD

        agg_float = agg_accumulator.astype(np.float32)
        agg_float[agg_float > (MOD // 2)] -= MOD
        agg_float /= SCALE
        
        update_dict = unflatten_params(agg_float, server.global_model)
        server_state = server.global_model.state_dict()
        for k in server_state.keys():
            if k in update_dict:
                server_state[k] += update_dict[k]
        
        server.update_global_model_with_state_dict(server_state)
        current_w_old_flat = flatten_params(server.global_model)

        acc, loss = server.evaluate()
        acc_history.append(acc)
        print(f"  [Round {r}] Global Acc: {acc:.2f}%, Loss: {loss:.4f} | Time: {time.time()-start_time:.1f}s")
        
        if atk_conf['active_attacks']:
            asr = server.evaluate_asr(test_loader, atk_conf['active_attacks'], atk_conf['attack_params'])
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