import os
import json
import numpy as np
import matplotlib.pyplot as plt
import datetime

def get_result_filename(mode_name, model_type, dataset_type, detection_method, config):
    """
    根据配置生成唯一的文件名标识
    格式: {mode}_{model}_{data}_{defense}_{attack}_{ratio}_{iid}
    """
    attack_type = "NoAttack"
    poison_ratio = config.get('poison_ratio', 0.0)
    
    # 获取攻击类型 (如果有)
    if poison_ratio > 0:
        # 尝试从 config['attack_types'] 或 config 中获取
        atks = config.get('attack_types', [])
        if not atks and 'attack' in config:
             atks = config['attack'].get('active_attacks', [])
        
        if atks:
            attack_type = "+".join(atks)
        else:
            attack_type = "UnknownAttack"
    
    iid_status = "NonIID" if config.get('if_noniid', False) else "IID"
    
    filename = f"{mode_name}_{model_type}_{dataset_type}_{detection_method}_{attack_type}_p{poison_ratio:.2f}_{iid_status}"
    return filename

def check_result_exists(save_dir, mode_name, model_type, dataset_type, detection_method, config):
    """检查结果文件是否已存在"""
    if not os.path.exists(save_dir):
        return False, None
        
    filename_base = get_result_filename(mode_name, model_type, dataset_type, detection_method, config)
    json_path = os.path.join(save_dir, f"{filename_base}_config.json")
    npz_path = os.path.join(save_dir, f"{filename_base}_data.npz")
    
    if os.path.exists(json_path) and os.path.exists(npz_path):
        try:
            with open(json_path, 'r') as f:
                saved_data = json.load(f)
            return True, saved_data
        except:
            return False, None
    return False, None

def save_result_with_config(save_dir, mode_name, model_type, dataset_type, detection_method, config, accuracy_history, asr_history=None, loss_history=None):
    """保存配置文件(JSON)和实验数据(NPZ)"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    filename_base = get_result_filename(mode_name, model_type, dataset_type, detection_method, config)
    
    # 1. 保存 Config (JSON)
    json_path = os.path.join(save_dir, f"{filename_base}_config.json")
    save_content = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": config,
        "results": {
            "final_accuracy": accuracy_history[-1] if accuracy_history else 0.0,
            "final_asr": asr_history[-1] if asr_history else 0.0,
            "max_accuracy": max(accuracy_history) if accuracy_history else 0.0
        },
        "accuracy_history": accuracy_history,
        "asr_history": asr_history if asr_history else [],
        "loss_history": loss_history if loss_history else []
    }
    
    # [修复] indent 参数移入 json.dump
    with open(json_path, 'w') as f:
        json.dump(save_content, f, indent=4)
        
    # 2. 保存数据 (NPZ)
    npz_path = os.path.join(save_dir, f"{filename_base}_data.npz")
    np.savez(
        npz_path, 
        accuracy=accuracy_history, 
        asr=asr_history if asr_history else [],
        loss=loss_history if loss_history else []
    )
    
    # 3. 绘图 (PNG)
    png_path = os.path.join(save_dir, f"{filename_base}_curve.png")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(accuracy_history)+1), accuracy_history, label='Main Task Accuracy', marker='o')
    if asr_history and len(asr_history) > 0 and max(asr_history) > 0:
        plt.plot(range(1, len(asr_history)+1), asr_history, label='Backdoor ASR', marker='x', linestyle='--')
    
    plt.title(f"Performance: {filename_base}")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Rate (%)")
    plt.grid(True)
    plt.legend()
    plt.savefig(png_path)
    plt.close()
    
    print(f"[Save] Result saved to {filename_base} (.json/.npz/.png)")

def plot_comparison_curves(config=None, result_dir="results", save_path="comparison.png"):
    """绘制对比曲线"""
    if not os.path.exists(result_dir):
        print(f"⚠️ 结果目录 {result_dir} 不存在。")
        return

    files = [f for f in os.listdir(result_dir) if f.endswith('.npz')]
    if not files:
        print(f"⚠️ 结果目录为空，跳过绘图")
        return
    
    if config:
        m_type = config.get('model_type', '')
        d_type = config.get('dataset_type', '')
        if m_type and d_type:
            target_token = f"{m_type}_{d_type}"
            files = [f for f in files if target_token in f]

    if not files:
        print("⚠️ 未找到匹配当前配置的结果文件。")
        return
    
    plt.figure(figsize=(12, 8))
    
    styles = {
        'pure_training': {'color': 'green', 'label': 'Benign (Baseline)', 'style': '--'},
        'poison_no_detection': {'color': 'red', 'label': 'Attack (No Defense)', 'style': '-'},
        'poison_with_detection': {'color': 'blue', 'label': 'Attack + Defense (Ours)', 'style': '-'}
    }
    
    has_data = False
    files.sort()

    for file in files:
        try:
            mode = None
            for k in styles.keys():
                if file.startswith(k):
                    mode = k
                    break
            
            if mode:
                data = np.load(os.path.join(result_dir, file), allow_pickle=True)
                acc_hist = data['accuracy_history']
                rounds = np.arange(1, len(acc_hist) + 1)
                
                style = styles[mode]
                
                # Accuracy 曲线
                plt.plot(rounds, acc_hist, 
                         color=style['color'], 
                         linestyle=style['style'], 
                         label=f"{style['label']} (Final Acc: {acc_hist[-1]:.1f}%)",
                         linewidth=2 if mode == 'poison_with_detection' else 1.5)
                
                has_data = True
                
        except Exception as e:
            print(f"Skip file {file}: {e}")

    if not has_data:
        print("⚠️ 找到文件但未匹配到任何已知模式。")
        return

    title = "Defensive Performance Comparison"
    if config:
        attack = config.get('attack_types', ['Unknown'])
        title += f"\nAttack: {attack} | Poison Ratio: {config.get('poison_ratio')} | { 'Non-IID' if config.get('if_noniid') else 'IID' }"
    
    plt.title(title)
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(save_path, dpi=300)
    print(f"📊 对比图已保存: {save_path}")
    plt.close()