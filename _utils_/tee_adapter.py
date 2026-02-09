import ctypes
import os
import numpy as np
import threading

_TEE_INSTANCE = None
_INIT_LOCK = threading.Lock()

def get_tee_adapter_singleton():
    global _TEE_INSTANCE
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
        self.lock = threading.Lock()
        
        self._load_library()
        self._init_functions()
        try:
            self.lib.tee_set_verbose.argtypes = [ctypes.c_int]
            self.lib.tee_set_verbose.restype = None
        except Exception:
            print("[Warning] tee_set_verbose not found in library.")
    
    def set_verbose(self, verbose_bool):
        level = 1 if verbose_bool else 0
        if self.initialized:
            self.lib.tee_set_verbose(level)

    def _load_library(self):
        if not os.path.exists(self.lib_path):
            raise FileNotFoundError(f"Lib not found: {self.lib_path}")
        self.lib = ctypes.CDLL(self.lib_path, mode=ctypes.RTLD_GLOBAL)

    def _to_bytes(self, val):
        return str(val).encode('utf-8')

    def _init_functions(self):
        self.lib.tee_init.argtypes = [ctypes.c_char_p]
        self.lib.tee_init.restype = ctypes.c_int
        self.lib.tee_destroy.argtypes = []
        
        # Phase 2: Prepare Gradient
        # Signature: (int cid, char* seed, float* w_new, float* w_old, size_t len, int* ranges, size_t r_len, float* out, size_t out_len)
        self.lib.tee_prepare_gradient.argtypes = [
            ctypes.c_int, 
            ctypes.c_char_p,
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
            ctypes.c_size_t,
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),
            ctypes.c_size_t,
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
            ctypes.c_size_t
        ]

        # Phase 4: Generate Masked Gradient
        # Signature: (char* root, char* g0, int cid, int* aids, size_t alen, char* kw, size_t len, int* ranges, size_t rlen, long* out, size_t olen)
        self.lib.tee_generate_masked_gradient_dynamic.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p,
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'), ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'), ctypes.c_size_t,
            np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS'), ctypes.c_size_t
        ]

        # Phase 5: Get Shares
        # Signature: (char* sss, char* root, int* u1, size_t l1, int* u2, size_t l2, int cid, int th, long* out, size_t max)
        self.lib.tee_get_vector_shares_dynamic.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p,
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'), ctypes.c_size_t,
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'), ctypes.c_size_t,
            ctypes.c_int, ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS'), ctypes.c_size_t
        ]

    def initialize_enclave(self, enclave_path=None):
        if self.initialized: return
        with self.lock:
            if self.initialized: return
            if enclave_path is None:
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                enclave_path = os.path.join(base_dir, "lib", "enclave.signed.so")
            if not os.path.exists(enclave_path): raise FileNotFoundError(f"Not found: {enclave_path}")
            self.enclave_path = enclave_path
            self.lib.tee_init(enclave_path.encode('utf-8'))
            self.initialized = True
            print(f"[TEEAdapter] Init OK: {enclave_path}")

    # --- Wrappers ---

    def prepare_gradient(self, client_id, proj_seed, w_new, w_old, output_dim=1024):
        """Phase 2: 计算梯度并投影，同时将梯度缓存于 TEE"""
        if not self.initialized: self.initialize_enclave()
        total_len = w_new.size
        # 如果模型参数本身已经是 float32，直接用；否则强转
        if w_new.dtype != np.float32: w_new = w_new.astype(np.float32)
        if w_old.dtype != np.float32: w_old = w_old.astype(np.float32)
        
        ranges = np.array([0, total_len], dtype=np.int32)
        output_proj = np.zeros(output_dim, dtype=np.float32)
        
        self.lib.tee_prepare_gradient(
            client_id, 
            self._to_bytes(proj_seed), 
            w_new, w_old, total_len, 
            ranges, len(ranges), 
            output_proj, output_dim
        )
        return output_proj, ranges

    def generate_masked_gradient_dynamic(self, seed_mask, seed_g0, cid, active_ids, k_weight, output_len):
        """Phase 4: 生成加掩码的梯度 (无需传入 w_new, 使用 TEE 缓存)"""
        if not self.initialized: self.initialize_enclave()
        
        model_len = output_len
        ranges = np.array([0, model_len], dtype=np.int32) # 默认全范围
        arr_active = np.array(active_ids, dtype=np.int32)
        out_buf = np.zeros(model_len, dtype=np.int64)
        
        self.lib.tee_generate_masked_gradient_dynamic(
            self._to_bytes(seed_mask), 
            self._to_bytes(seed_g0), 
            cid, 
            arr_active, len(arr_active), 
            self._to_bytes(k_weight), 
            model_len, 
            ranges, len(ranges), 
            out_buf, model_len
        )
        return out_buf

    def get_vector_shares_dynamic(self, seed_sss, seed_mask, u1_ids, u2_ids, my_cid, threshold):
        """Phase 5: 获取秘密分片 (Shares)"""
        if not self.initialized: self.initialize_enclave()
        arr_u1 = np.array(u1_ids, dtype=np.int32)
        arr_u2 = np.array(u2_ids, dtype=np.int32)
        
        # Share 0: Delta (Needs Reconstruction)
        # Share 1: Alpha Seed (Needs Reconstruction)
        # Share 2...N: Beta Seeds (For each online client)
        # Buffer size: 2 + len(u2)
        max_len = 2 + len(u2_ids)
        out_buf = np.zeros(max_len, dtype=np.int64)
        
        self.lib.tee_get_vector_shares_dynamic(
            self._to_bytes(seed_sss), 
            self._to_bytes(seed_mask), 
            arr_u1, len(arr_u1), 
            arr_u2, len(arr_u2), 
            my_cid, threshold, 
            out_buf, max_len
        )
        return out_buf