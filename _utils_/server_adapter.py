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
        
        # Calc Secrets
        self.lib.server_core_calc_secrets.argtypes = [
            ctypes.c_char_p, # seed_mask_root_str
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'), ctypes.c_int, # u1
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'), ctypes.c_int, # u2
            ctypes.c_char_p, # out_delta_str
            ctypes.c_char_p, # out_n_sum_str
            ctypes.c_size_t  # buffer_len
        ]
        
        # Gen Noise
        self.lib.server_core_gen_noise_vector.argtypes = [
            ctypes.c_char_p, # seed_root
            ctypes.c_char_p, # seed_g0
            ctypes.c_char_p, # delta
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'), ctypes.c_int, # u2
            np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS'), # output noise
            ctypes.c_int   # len
        ]

    def _to_bytes(self, val):
        return str(val).encode('utf-8')

    def calculate_secrets(self, seed_mask_root, u1_ids, u2_ids):
        arr_u1 = np.array(u1_ids, dtype=np.int32)
        arr_u2 = np.array(u2_ids, dtype=np.int32)
        
        buf_len = 64
        d_buf = ctypes.create_string_buffer(buf_len)
        n_buf = ctypes.create_string_buffer(buf_len)
        
        self.lib.server_core_calc_secrets(
            self._to_bytes(seed_mask_root),
            arr_u1, len(arr_u1),
            arr_u2, len(arr_u2),
            d_buf,
            n_buf,
            buf_len
        )
        try:
            delta = int(d_buf.value.decode('utf-8'))
            n_sum = int(n_buf.value.decode('utf-8'))
            return delta, n_sum
        except ValueError:
            return 0, 0

    def generate_noise_vector(self, seed_mask_root, seed_global_0, delta, u2_ids, data_len):
        arr_u2 = np.array(u2_ids, dtype=np.int32)
        out_noise = np.zeros(data_len, dtype=np.int64)
        
        self.lib.server_core_gen_noise_vector(
            self._to_bytes(seed_mask_root),
            self._to_bytes(seed_global_0),
            self._to_bytes(delta),
            arr_u2, len(arr_u2),
            out_noise,
            data_len
        )
        return out_noise