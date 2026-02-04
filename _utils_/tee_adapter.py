# _utils_/tee_adapter.py
import ctypes
import os
import numpy as np
import torch

class TEEAdapter:
    def __init__(self, lib_path=None):
        if lib_path is None:
            # 默认寻找项目根目录下的 lib/libtee_bridge.so
            # 假设当前文件在 _utils_ 目录下，向上两级找到 lib
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            lib_path = os.path.join(base_dir, "lib", "libtee_bridge.so")

        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"TEE Bridge library not found at: {lib_path}")

        print(f"[TEEAdapter] Loading SGX bridge from: {lib_path}")
        self.lib = ctypes.CDLL(lib_path)
        
        # 1. 初始化接口
        self.lib.tee_init.argtypes = [ctypes.c_char_p]
        self.lib.tee_init.restype = ctypes.c_int
        
        # 2. 销毁接口
        self.lib.tee_destroy.argtypes = []
        self.lib.tee_destroy.restype = None
        
        # 3. 核心安全聚合接口
        # int tee_secure_aggregation(long seed, float* w_new, float* w_old, int model_len, int* ranges, int ranges_len, float* output, int out_len)
        self.lib.tee_secure_aggregation.argtypes = [
            ctypes.c_long,                                                  # seed
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'), # w_new
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'), # w_old
            ctypes.c_int,                                                   # model_len
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),   # ranges
            ctypes.c_int,                                                   # ranges_len
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'), # output
            ctypes.c_int                                                    # out_len
        ]
        self.lib.tee_secure_aggregation.restype = ctypes.c_int

    def initialize_enclave(self, enclave_path=None):
        if enclave_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            enclave_path = os.path.join(base_dir, "lib", "enclave.signed.so")
            
        if not os.path.exists(enclave_path):
            raise FileNotFoundError(f"Signed Enclave not found at: {enclave_path}")

        # 必须转为 bytes 传给 C
        ret = self.lib.tee_init(enclave_path.encode('utf-8'))
        if ret != 0:
            raise RuntimeError("Failed to initialize SGX Enclave!")
        print("[TEEAdapter] Enclave initialized successfully.")

    def secure_project(self, seed, w_new_flat, w_old_flat, ranges, output_dim=1024):
        """
        执行 TEE 内的梯度计算与投影
        :param seed: 投影种子 (int)
        :param w_new_flat: 训练后参数 (numpy float32 array)
        :param w_old_flat: 训练前参数 (numpy float32 array)
        :param ranges: 切片区间列表 [[start, len], [start, len]...]
        :param output_dim: 输出摘要长度
        :return: 投影摘要 (numpy float32 array)
        """
        # 数据检查
        if w_new_flat.shape != w_old_flat.shape:
            raise ValueError("Shape mismatch between w_new and w_old")
        
        total_len = w_new_flat.size
        
        # 处理 Ranges
        # 将 [[s, l], [s, l]] 展平为 [s, l, s, l]
        if not ranges:
            # 默认为全量: [0, total_len]
            ranges_flat = np.array([0, total_len], dtype=np.int32)
        else:
            ranges_flat = np.array(ranges, dtype=np.int32).flatten()
            
        # 准备输出容器
        output = np.zeros(output_dim, dtype=np.float32)
        
        # 调用 C 接口
        ret = self.lib.tee_secure_aggregation(
            seed,
            w_new_flat,
            w_old_flat,
            total_len,
            ranges_flat,
            len(ranges_flat),
            output,
            output_dim
        )
        
        if ret != 0:
            raise RuntimeError("TEE secure aggregation failed!")
            
        return output

    def close(self):
        self.lib.tee_destroy()