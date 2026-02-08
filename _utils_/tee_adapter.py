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
        
        # Prepare Gradient
        try: func = self.lib.tee_prepare_gradient
        except: func = self.lib.ecall_prepare_gradient
        func.argtypes = [
            ctypes.c_int, 
            ctypes.c_char_p, # proj_seed_str
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
            ctypes.c_int
        ]
        self._func_prepare = func

        # Generate Masked
        try: func = self.lib.tee_generate_masked_gradient_dynamic
        except: func = self.lib.ecall_generate_masked_gradient_dynamic
        func.argtypes = [
            ctypes.c_char_p, # seed_root
            ctypes.c_char_p, # seed_g0
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'), ctypes.c_int,
            ctypes.c_char_p, # k_weight (float string)
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'), ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS'), ctypes.c_int
        ]
        self._func_generate = func

        # Get Shares
        try: func = self.lib.tee_get_vector_shares_dynamic
        except: func = self.lib.ecall_get_vector_shares_dynamic
        func.argtypes = [
            ctypes.c_char_p, # seed_sss
            ctypes.c_char_p, # seed_root
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'), ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'), ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS'), ctypes.c_int
        ]
        self._func_get_shares = func

        # Noise Gen
        try: func = self.lib.tee_generate_noise_from_seed
        except: func = self.lib.ecall_generate_noise_from_seed
        func.argtypes = [
            ctypes.c_char_p, # seed
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
        ]
        self._func_noise = func

        # --- Server Core ---
        try:
            func = self.lib.tee_server_calculate_secrets
            func.argtypes = [
                ctypes.c_char_p,
                np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'), ctypes.c_int,
                np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'), ctypes.c_int,
                ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t
            ]
            self._func_server_calc = func
            
            func = self.lib.tee_server_gen_noise_vector
            func.argtypes = [
                ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'), ctypes.c_int,
                np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS'), ctypes.c_int
            ]
            self._func_server_noise = func
        except: pass

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
        if not self.initialized: self.initialize_enclave()
        total_len = w_new.size
        ranges = np.array([0, total_len], dtype=np.int32)
        output_proj = np.zeros(output_dim, dtype=np.float32)
        self._func_prepare(
            client_id, 
            self._to_bytes(proj_seed), # Str
            w_new, w_old, total_len, ranges, len(ranges), output_proj, output_dim
        )
        return output_proj, ranges

    def generate_masked_gradient_dynamic(self, seed_mask, seed_g0, cid, active_ids, k_weight, w_new, ranges, output_len):
        if not self.initialized: self.initialize_enclave()
        model_len = output_len
        if ranges is None: ranges = np.array([0, model_len], dtype=np.int32)
        arr_active = np.array(active_ids, dtype=np.int32)
        out_buf = np.zeros(model_len, dtype=np.int64)
        
        self._func_generate(
            self._to_bytes(seed_mask), # Str
            self._to_bytes(seed_g0),   # Str
            cid, 
            arr_active, len(arr_active), 
            self._to_bytes(k_weight),  # Float -> Str
            model_len, ranges, len(ranges), 
            out_buf, model_len
        )
        return out_buf

    def get_vector_shares_dynamic(self, seed_sss, seed_mask, u1_ids, u2_ids, my_cid, threshold):
        if not self.initialized: self.initialize_enclave()
        arr_u1 = np.array(u1_ids, dtype=np.int32)
        arr_u2 = np.array(u2_ids, dtype=np.int32)
        max_len = 2 + len(u2_ids) + 50
        out_buf = np.zeros(max_len, dtype=np.int64)
        
        self._func_get_shares(
            self._to_bytes(seed_sss),  # Str
            self._to_bytes(seed_mask), # Str
            arr_u1, len(arr_u1), arr_u2, len(arr_u2), 
            my_cid, threshold, out_buf, max_len
        )
        return out_buf[:2 + len(u2_ids)]

    # --- Server Side Wrappers ---
    
    def server_calculate_secrets(self, seed_mask_root, u1_ids, u2_ids):
        arr_u1 = np.array(u1_ids, dtype=np.int32)
        arr_u2 = np.array(u2_ids, dtype=np.int32)
        buf_len = 64
        d_buf = ctypes.create_string_buffer(buf_len)
        n_buf = ctypes.create_string_buffer(buf_len)
        
        self._func_server_calc(
            self._to_bytes(seed_mask_root),
            arr_u1, len(arr_u1), arr_u2, len(arr_u2),
            d_buf, n_buf, buf_len
        )
        return int(d_buf.value), int(n_buf.value)

    def server_generate_noise_vector(self, seed_mask_root, seed_global_0, delta, u2_ids, data_len):
        arr_u2 = np.array(u2_ids, dtype=np.int32)
        out_noise = np.zeros(data_len, dtype=np.int64)
        
        self._func_server_noise(
            self._to_bytes(seed_mask_root),
            self._to_bytes(seed_global_0),
            self._to_bytes(delta),
            arr_u2, len(arr_u2),
            out_noise, data_len
        )
        return out_noise