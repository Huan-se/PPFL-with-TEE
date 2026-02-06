import ctypes
import os
import numpy as np

# 注意：之前的 SharePackage 结构体已不再需要，因为我们现在返回扁平的 long long 数组

class TEEAdapter:
    def __init__(self, lib_path=None):
        if lib_path is None:
            # 默认寻找项目根目录下的 lib/libtee_bridge.so
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            lib_path = os.path.join(base_dir, "lib", "libtee_bridge.so")
        
        self.lib_path = lib_path
        self.lib = None
        self.enclave_path = None
        self.initialized = False
        
        # 加载库
        self._load_library()
        try:
            func_noise = self.lib.tee_generate_noise_from_seed
        except AttributeError:
            func_noise = self.lib.ecall_generate_noise_from_seed

        func_noise.argtypes = [
            ctypes.c_long,  # seed
            ctypes.c_int,   # len
            np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS') # output
        ]
        func_noise.restype = None
        self._func_noise = func_noise

    def _load_library(self):
        """加载 .so 库并配置参数类型"""
        if not os.path.exists(self.lib_path):
            raise FileNotFoundError(f"TEE Bridge library not found at: {self.lib_path}")

        self.lib = ctypes.CDLL(self.lib_path)
        
        # 1. 初始化接口
        self.lib.tee_init.argtypes = [ctypes.c_char_p]
        self.lib.tee_init.restype = ctypes.c_int
        
        # 2. 销毁接口
        self.lib.tee_destroy.argtypes = []
        self.lib.tee_destroy.restype = None
        
        # =========================================================
        # [Phase 2] 准备梯度
        # =========================================================
        try:
            func_prep = self.lib.tee_prepare_gradient
        except AttributeError:
            func_prep = self.lib.ecall_prepare_gradient
            
        func_prep.argtypes = [
            ctypes.c_int,   # client_id
            ctypes.c_long,  # proj_seed
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'), # w_new
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'), # w_old
            ctypes.c_int,   # model_len
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),   # ranges
            ctypes.c_int,   # ranges_len
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'), # output_proj
            ctypes.c_int    # out_len
        ]
        func_prep.restype = None
        self._func_prepare = func_prep

        # =========================================================
        # [Phase 4] 生成双掩码梯度 (参数已更新)
        # =========================================================
        try:
            func_mask = self.lib.tee_generate_masked_gradient_dynamic
        except AttributeError:
            func_mask = self.lib.ecall_generate_masked_gradient_dynamic
            
        func_mask.argtypes = [
            ctypes.c_long,  # seed_mask_root
            ctypes.c_long,  # seed_global_0
            ctypes.c_int,   # client_id
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),   # active_ids
            ctypes.c_int,   # active_count
            ctypes.c_float, # k_weight
            ctypes.c_int,   # model_len
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),   # ranges
            ctypes.c_int,   # ranges_len
            np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS'),   # output (int64/long long)
            ctypes.c_int    # out_len
        ]
        func_mask.restype = None
        self._func_generate = func_mask

        # =========================================================
        # [Phase 5] 获取秘密分片 (参数已更新)
        # =========================================================
        try:
            func_sss = self.lib.tee_get_vector_shares_dynamic
        except AttributeError:
            func_sss = self.lib.ecall_get_vector_shares_dynamic

        func_sss.argtypes = [
            ctypes.c_long,  # seed_sss
            ctypes.c_long,  # seed_mask_root
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'), # u1_ids
            ctypes.c_int,   # u1_len
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'), # u2_ids
            ctypes.c_int,   # u2_len
            ctypes.c_int,   # my_client_id
            ctypes.c_int,   # threshold
            np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS'), # output_vector
            ctypes.c_int    # out_max_len
        ]
        func_sss.restype = None
        self._func_get_shares = func_sss

    def initialize_enclave(self, enclave_path=None):
        if self.initialized: return
        if enclave_path is None:
            if self.enclave_path:
                enclave_path = self.enclave_path
            else:
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                enclave_path = os.path.join(base_dir, "lib", "enclave.signed.so")
        if not os.path.exists(enclave_path):
            raise FileNotFoundError(f"Signed Enclave not found at: {enclave_path}")
        self.enclave_path = enclave_path
        self.lib.tee_init(enclave_path.encode('utf-8'))
        self.initialized = True

    # =========================================================
    # Python 调用接口
    # =========================================================

    def prepare_gradient(self, client_id, proj_seed, w_new, w_old, output_dim=1024):
        if not self.initialized: self.initialize_enclave()
        total_len = w_new.size
        ranges = np.array([0, total_len], dtype=np.int32)
        output_proj = np.zeros(output_dim, dtype=np.float32)
        self._func_prepare(client_id, proj_seed, w_new, w_old, total_len, ranges, len(ranges), output_proj, output_dim)
        return output_proj

    def generate_masked_gradient(self, seed_mask, seed_g0, cid, active_ids, k_weight, model_len=0, ranges=None):
        """
        [Phase 4] 调用 TEE 生成加密梯度
        Args:
            active_ids: List[int], 本轮参与聚合的客户端 ID 列表 (用于计算互掩码)
            k_weight: float, 服务端分配的聚合权重
        """
        if not self.initialized: self.initialize_enclave()
        if ranges is None: 
            ranges = np.array([0, model_len], dtype=np.int32)
        
        # 转换 active_ids 为 numpy 数组
        arr_active = np.array(active_ids, dtype=np.int32)
        
        out_buf = np.zeros(model_len, dtype=np.int64)
        
        self._func_generate(
            seed_mask, seed_g0, cid, 
            arr_active, len(arr_active), 
            k_weight, 
            model_len, ranges, len(ranges), 
            out_buf, model_len
        )
        return out_buf
    def generate_noise_from_seed(self, seed, length):
        """[Server Helper] 使用指定种子生成掩码序列"""
        if not self.initialized: self.initialize_enclave()
        out_buf = np.zeros(length, dtype=np.int64)
        self._func_noise(seed, length, out_buf)
        return out_buf

    def get_vector_shares(self, seed_sss, seed_mask, u1_ids, u2_ids, my_cid, threshold):
        """
        [Phase 5] 计算掉线恢复秘密分片
        Args:
            u1_ids: List[int], Phase 4 参与者列表 (用于计算 Inv)
            u2_ids: List[int], Phase 5 存活者列表 (用于计算 Delta)
        Returns:
            np.array[int64]: 秘密分片向量
        """
        if not self.initialized: self.initialize_enclave()
        
        arr_u1 = np.array(u1_ids, dtype=np.int32)
        arr_u2 = np.array(u2_ids, dtype=np.int32)
        
        # 预估最大输出长度: Delta(1) + Alpha(1) + Max_Clients
        max_len = 2 + len(u2_ids) + 10 
        out_buf = np.zeros(max_len, dtype=np.int64)
        
        self._func_get_shares(
            seed_sss, seed_mask, 
            arr_u1, len(arr_u1), 
            arr_u2, len(arr_u2), 
            my_cid, threshold, 
            out_buf, max_len
        )
        
        # TEE 会将有效数据后面填充 0，但 Python 侧暂时返回整个 buffer
        # 实际有效长度由 Server 重构时的逻辑决定，或者在这里根据 U2 长度截断
        # 秘密向量长度 = 2 + len(u2_ids)
        actual_len = 2 + len(u2_ids)
        return out_buf[:actual_len]