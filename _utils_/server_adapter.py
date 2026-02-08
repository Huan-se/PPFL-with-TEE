import ctypes
import os
import numpy as np

class ServerAdapter:
    def __init__(self, lib_path=None):
        if lib_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            lib_path = os.path.join(base_dir, "lib", "libserver_core.so")
        
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Server Core library not found at: {lib_path}")

        self.lib = ctypes.CDLL(lib_path)
        
        # 配置 C++ 接口参数类型
        # void server_core_aggregate_and_unmask(
        #     const char* seed_root, const char* seed_g0, 
        #     int* u1, int l1, int* u2, int l2, 
        #     long* shares, int vec_len, 
        #     long* ciphers, int data_len, 
        #     long* output
        # )
        self.lib.server_core_aggregate_and_unmask.argtypes = [
            ctypes.c_char_p, # seed_mask_root
            ctypes.c_char_p, # seed_global_0
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'), ctypes.c_int, # u1_ids
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'), ctypes.c_int, # u2_ids
            np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS'), # shares_flat
            ctypes.c_int, # vector_len
            np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS'), # ciphers_flat
            ctypes.c_int, # data_len
            np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')  # output_result
        ]

    # [关键修复] 补全缺失的辅助函数
    def _to_bytes(self, val):
        return str(val).encode('utf-8')

    def aggregate_and_unmask(self, seed_mask_root, seed_global_0, u1_ids, u2_ids, shares_list, cipher_list):
        """
        一站式安全聚合：
        1. 批量恢复 Secret 向量 (Delta, Alpha, Beta...)
        2. 生成 Noise
        3. 聚合并消去
        """
        # 确保输入数据非空
        if not cipher_list or not shares_list:
            raise ValueError("Cipher list or Shares list is empty!")

        data_len = len(cipher_list[0])
        vector_len = len(shares_list[0]) # 每个 Client 的 Share 向量长度
        num_u1 = len(u1_ids)
        num_u2 = len(u2_ids)
        
        arr_u1 = np.array(u1_ids, dtype=np.int32)
        arr_u2 = np.array(u2_ids, dtype=np.int32)
        
        # 展平 Shares: [Client1_Vec, Client2_Vec...] -> 1D Array
        # shares_list 是一个 list of lists/arrays，vstack 将其堆叠为矩阵，再 reshape 展平
        flat_shares = np.vstack(shares_list).reshape(-1).astype(np.int64)
        
        # 展平 Ciphers
        flat_ciphers = np.vstack(cipher_list).reshape(-1).astype(np.int64)
        
        # 准备输出缓冲区
        out_result = np.zeros(data_len, dtype=np.int64)
        
        # 调用 C++
        self.lib.server_core_aggregate_and_unmask(
            self._to_bytes(seed_mask_root),
            self._to_bytes(seed_global_0),
            arr_u1, num_u1,
            arr_u2, num_u2,
            flat_shares,
            vector_len,
            flat_ciphers,
            data_len,
            out_result
        )
        
        return out_result