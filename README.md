# PPFL-with-TEE: Secure Federated Learning with Poisoning Defense

本项目实现了一个基于 **Intel SGX (TEE)** 的安全联邦学习系统，集成了隐私保护聚合与投毒攻击防御机制。

## 1. 核心功能

* **🛡️ TEE 安全聚合 (Secure Aggregation)**
* 利用 Intel SGX Enclave 在受信执行环境中进行梯度加权与聚合。
* 服务器无法窥探单个客户端的原始梯度，仅能获得聚合后的全局更新。
* 支持基于秘密共享 (Shamir's Secret Sharing) 的客户端掉线恢复机制。


* **🔍 投毒防御 (Poisoning Defense)**
* **算法**: 基于层级投影 (Layer-wise Projection) 的异常检测 (参考 MESAS)。
* **机制**: 在 TEE 外部根据梯度特征（L2 范数、角度距离）计算异常分数，动态调整聚合权重或剔除恶意客户端。


* **⚔️ 攻击模拟 (Attack Simulation)**
* 支持 **Backdoor** (后门攻击) 和 **Label Flip** (标签翻转)。
* 提供严谨的 ASR (攻击成功率) 评估：在评估阶段强制 100% 投毒以测得真实攻击底线。



## 2. 环境依赖

### 硬件要求

* **CPU**: 支持 Intel SGX 的处理器（若无硬件，需使用模拟模式编译）。
* **BIOS**: 需开启 SGX (Software Controlled 或 Enabled)。

### 软件要求

* **OS**: Ubuntu 18.04 / 20.04 / 22.04 (推荐 Linux Kernel 5.11+ 以支持 SGX In-Kernel Driver)。
* **Intel SGX SDK**: [Intel SGX SDK for Linux](https://github.com/intel/linux-sgx).
* **Python**: 3.8+
* **Python 库**:
```bash
pip install torch torchvision numpy pyyaml matplotlib scikit-learn

```



## 3. 编译指南

本项目包含 C++ 核心库（Enclave & Bridge），运行前**必须**编译。

1. **加载 SGX 环境** (根据实际安装路径调整):
```bash
source /opt/intel/sgxsdk/environment

```


2. **编译动态库**:
* **硬件模式 (推荐，需 SGX 硬件)**:
```bash
cd PPFL-with-TEE-main
make clean && make SGX_MODE=HW

```


* **模拟模式 (无硬件调试用)**:
```bash
cd PPFL-with-TEE-main
make clean && make SGX_MODE=SIM

```




*编译成功后，`lib/` 目录下应生成 `libtee_bridge.so`, `enclave.signed.so`, `libserver_core.so`。*

## 4. 运行与测试

所有实验通过 `main/main.py` 入口启动，支持三种标准模式。

### 模式 A: 基础训练 (基准测试)

验证联邦学习流程是否通畅，模型是否收敛。

```bash
python main/main.py --mode pure_training

```

* **预期**: `ACC` 稳步上升，`ASR` 应接近 0%。

### 模式 B: 攻击测试 (无防御)

验证投毒攻击是否生效。

```bash
python main/main.py --mode poison_no_detection

```

* **预期**: `ACC` 正常或略降，`ASR` (攻击成功率) 显著上升 (如 >80%)。

### 模式 C: 防御测试 (开启检测)

验证防御算法是否能检测并剔除恶意客户端。

```bash
python main/main.py --mode poison_with_detection

```

* **预期**: `ACC` 恢复正常，`ASR` 被压制在低水平 (如 <5%)。

## 5. 结果与日志

实验结果将保存在 `main/results/` 目录下：

1. **`*_curve.png`**: 包含 Accuracy 和 ASR 的变化曲线图。
2. **`*_detection_log.csv` (核心调试)**:
* 记录每一轮防御算法的详细判定数据。
* 字段: `Client_ID`, `Type` (Malicious/Benign), `Score` (异常分), `Status` (Suspect/Normal)。
* **用途**: 检查恶意客户端是否被标记为 `SUSPECT`，以及权重是否被置 0。


3. **`*_config.json`**: 实验配置快照及最终结果摘要。

## 6. 常用配置说明

修改 `config/config.yaml` 可调整实验参数：

```yaml
experiment:
  verbose: false          # [重要] 设为 true 可开启 C++ 底层详细日志(Debug用)，平时建议 false 以保持清爽
  device: "cuda"          # 训练设备

federated:
  comm_rounds: 50         # 通信轮次
  total_clients: 20       # 客户端总数

attack:
  poison_ratio: 0.2       # 恶意客户端比例 (如 20%)
  active_attacks: ["backdoor"] # 攻击类型: backdoor 或 label_flip

defense:
  method: "layers_proj_detect" # 推荐防御方法

```
