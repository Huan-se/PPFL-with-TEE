import ctypes
import os
import numpy as np
import threading  # [新增]

# 全局单例存储
_TEE_INSTANCE = None
# 全局锁，防止多线程同时创建 Adapter 或 Enclave
_INIT_LOCK = threading.Lock()

def get_tee_adapter_singleton():
    """获取全局唯一的 TEEAdapter 实例"""
    global _TEE_INSTANCE
    # 双重检查锁定 (Double-Checked Locking)
    if _TEE_INSTANCE is None:
        with _INIT_LOCK:
            if _TEE_INSTANCE is None:
                _TEE_INSTANCE = TEEAdapter()
    return _TEE_INSTANCE

class TEEAdapter:
    def __init__(self, lib_path=None):
        if lib_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            lib_path = os.path.join(base_dir, "lib", "libtee_bridge.so")
        
        self.lib_path = lib_path
        self.lib = None
        self.enclave_path = None
        self.initialized = False
        self.lock = threading.Lock()  # [新增] 实例级锁
        
        self._load_library()
        self._init_functions()

    def _load_library(self):
        if not os.path.exists(self.lib_path):
            raise FileNotFoundError(f"TEE Bridge library not found at: {self.lib_path}")
        # 使用 RTLD_GLOBAL 确保符号全局可见，避免重复加载问题
        self.lib = ctypes.CDLL(self.lib_path, mode=ctypes.RTLD_GLOBAL)

    def _init_functions(self):
        # 1. Init / Destroy
        self.lib.tee_init.argtypes = [ctypes.c_char_p]
        self.lib.tee_init.restype = ctypes.c_int
        self.lib.tee_destroy.argtypes = []
        self.lib.tee_destroy.restype = None

        # 2. Phase 2: Prepare
        try: func_prep = self.lib.tee_prepare_gradient
        except AttributeError: func_prep = self.lib.ecall_prepare_gradient
        func_prep.argtypes = [
            ctypes.c_int, ctypes.c_long,
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
            ctypes.c_int
        ]
        func_prep.restype = None
        self._func_prepare = func_prep

        # 3. Phase 4: Generate Masked
        try: func_mask = self.lib.tee_generate_masked_gradient_dynamic
        except AttributeError: func_mask = self.lib.ecall_generate_masked_gradient_dynamic
        func_mask.argtypes = [
            ctypes.c_long, ctypes.c_long, ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'), ctypes.c_int,
            ctypes.c_float, ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'), ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS'), ctypes.c_int
        ]
        func_mask.restype = None
        self._func_generate = func_mask

        # 4. Phase 5: Get Shares
        try: func_sss = self.lib.tee_get_vector_shares_dynamic
        except AttributeError: func_sss = self.lib.ecall_get_vector_shares_dynamic
        func_sss.argtypes = [
            ctypes.c_long, ctypes.c_long,
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'), ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'), ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS'), ctypes.c_int
        ]
        func_sss.restype = None
        self._func_get_shares = func_sss

        # 5. Noise Gen
        try: func_noise = self.lib.tee_generate_noise_from_seed
        except AttributeError: func_noise = self.lib.ecall_generate_noise_from_seed
        func_noise.argtypes = [
            ctypes.c_long, ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
        ]
        func_noise.restype = None
        self._func_noise = func_noise

    def initialize_enclave(self, enclave_path=None):
        """线程安全的初始化函数"""
        # 快速检查
        if self.initialized: return
        
        # 加锁初始化，防止并发创建多个 Enclave
        with self.lock:
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
            # 调用 C 库初始化
            self.lib.tee_init(enclave_path.encode('utf-8'))
            self.initialized = True
            print(f"[TEEAdapter] Enclave initialized successfully (Thread Safe). Path: {enclave_path}")

    # --- Wrapper Methods ---

    def prepare_gradient(self, client_id, proj_seed, w_new, w_old, output_dim=1024):
        if not self.initialized: self.initialize_enclave()
        total_len = w_new.size
        ranges = np.array([0, total_len], dtype=np.int32)
        output_proj = np.zeros(output_dim, dtype=np.float32)
        
        self._func_prepare(client_id, proj_seed, w_new, w_old, total_len, ranges, len(ranges), output_proj, output_dim)
        return output_proj, ranges

    def generate_masked_gradient_dynamic(self, seed_mask, seed_g0, cid, active_ids, k_weight, w_new, ranges, output_len):
        if not self.initialized: self.initialize_enclave()
        
        model_len = output_len
        if ranges is None: ranges = np.array([0, model_len], dtype=np.int32)
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
        if not self.initialized: self.initialize_enclave()
        out_buf = np.zeros(length, dtype=np.int64)
        self._func_noise(seed, length, out_buf)
        return out_buf

    def get_vector_shares_dynamic(self, seed_sss, seed_mask, u1_ids, u2_ids, my_cid, threshold):
        if not self.initialized: self.initialize_enclave()
        arr_u1 = np.array(u1_ids, dtype=np.int32)
        arr_u2 = np.array(u2_ids, dtype=np.int32)
        max_len = 2 + len(u2_ids) + 50 
        out_buf = np.zeros(max_len, dtype=np.int64)
        
        self._func_get_shares(
            seed_sss, seed_mask, arr_u1, len(arr_u1), arr_u2, len(arr_u2), 
            my_cid, threshold, out_buf, max_len
        )
        return out_buf[:2 + len(u2_ids)]