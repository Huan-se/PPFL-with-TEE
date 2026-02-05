import ctypes
import os
import numpy as np

# 定义与 Enclave/App 中一致的结构体
# 用于接收打包的秘密分片 (Phase 5)
class SharePackage(ctypes.Structure):
    _fields_ = [
        ("share_alpha", ctypes.c_float), # 对应 Enclave 中的 share_alpha
        ("share_beta", ctypes.c_float)   # 对应 Enclave 中的 share_beta
    ]

class TEEAdapter:
    def __init__(self, lib_path=None):
        if lib_path is None:
            # 默认寻找项目根目录下的 lib/libtee_bridge.so
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            lib_path = os.path.join(base_dir, "lib", "libtee_bridge.so")

        self.lib_path = lib_path
        self.lib = None
        
        # 保存 Enclave 路径以便在子进程中自动重连
        self.enclave_path = None 
        # 标记当前进程是否已初始化 Enclave
        self.initialized = False 
        
        # 加载库
        self._load_library()

    def _load_library(self):
        """加载 .so 库并配置参数类型"""
        if not os.path.exists(self.lib_path):
            raise FileNotFoundError(f"TEE Bridge library not found at: {self.lib_path}")

        # 加载动态库
        self.lib = ctypes.CDLL(self.lib_path)
        
        # 1. 初始化接口
        self.lib.tee_init.argtypes = [ctypes.c_char_p]
        self.lib.tee_init.restype = ctypes.c_int
        
        # 2. 销毁接口
        self.lib.tee_destroy.argtypes = []
        self.lib.tee_destroy.restype = None
        
        # 3. [Phase 2] 投影接口 (LSH)
        # int tee_secure_aggregation_phase(...)
        self.lib.tee_secure_aggregation_phase.argtypes = [
            ctypes.c_long,                                                  # seed
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'), # w_new
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'), # w_old
            ctypes.c_int,                                                   # model_len
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),   # ranges
            ctypes.c_int,                                                   # ranges_len
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'), # output
            ctypes.c_int                                                    # out_len
        ]
        self.lib.tee_secure_aggregation_phase.restype = ctypes.c_int

        # 4. [Phase 4] 双掩码梯度生成接口 (Dynamic Derivation)
        # int ecall_generate_masked_gradient_dynamic(...)
        # 注意：这里调用的是 App.cpp 导出的接口名。
        # 如果 App.cpp 中使用了 tee_ 前缀，请确保名字一致。
        # 这里假设 App.cpp 导出的名字直接对应 EDL 的功能名，或者叫 tee_generate_masked_gradient_dynamic
        # 为了稳妥，通常我们在 App.cpp 会加 tee_ 前缀。这里假设你会在 App.cpp 导出时命名为 tee_generate_masked_gradient_dynamic
        try:
            func = self.lib.tee_generate_masked_gradient_dynamic
        except AttributeError:
            # 回退尝试没有 tee_ 前缀的情况 (取决于 App.cpp 怎么写)
            func = self.lib.ecall_generate_masked_gradient_dynamic
            
        func.argtypes = [
            ctypes.c_long,  # seed_mask_root
            ctypes.c_long,  # seed_global_0
            ctypes.c_int,   # client_id
            ctypes.c_float, # k_weight
            ctypes.c_float, # n_ratio
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'), # w_new
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'), # w_old
            ctypes.c_int,   # model_len
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),   # ranges
            ctypes.c_int,   # ranges_len
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'), # output (buffer for int32)
            ctypes.c_int    # out_len
        ]
        func.restype = ctypes.c_int
        self._func_generate_masked = func

        # 5. [Phase 5] 向量化秘密共享接口
        # int tee_get_vector_shares_dynamic(...)
        try:
            func_sss = self.lib.tee_get_vector_shares_dynamic
        except AttributeError:
            func_sss = self.lib.ecall_get_vector_shares_dynamic

        func_sss.argtypes = [
            ctypes.c_long,  # seed_sss
            ctypes.c_long,  # seed_mask_root
            ctypes.c_int,   # target_client_id
            ctypes.c_int,   # threshold
            ctypes.c_int,   # total_clients
            ctypes.POINTER(SharePackage) # output_shares array
        ]
        func_sss.restype = ctypes.c_int
        self._func_get_shares = func_sss

    def __getstate__(self):
        """序列化前调用：移除不可 pickle 的 ctypes 对象"""
        state = self.__dict__.copy()
        if 'lib' in state:
            del state['lib']
        if '_func_generate_masked' in state:
            del state['_func_generate_masked']
        if '_func_get_shares' in state:
            del state['_func_get_shares']
            
        state['initialized'] = False 
        return state

    def __setstate__(self, state):
        """反序列化后调用：重新加载动态库"""
        self.__dict__.update(state)
        self._load_library() 

    def initialize_enclave(self, enclave_path=None):
        """初始化 Enclave"""
        if self.initialized:
            return 

        if enclave_path is None:
            if self.enclave_path:
                enclave_path = self.enclave_path
            else:
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                enclave_path = os.path.join(base_dir, "lib", "enclave.signed.so")
            
        if not os.path.exists(enclave_path):
            raise FileNotFoundError(f"Signed Enclave not found at: {enclave_path}")

        self.enclave_path = enclave_path

        ret = self.lib.tee_init(enclave_path.encode('utf-8'))
        if ret != 0:
            raise RuntimeError(f"Failed to initialize SGX Enclave (PID: {os.getpid()})")
            
        self.initialized = True

    def secure_project(self, seed, w_new_flat, w_old_flat, ranges, output_dim=1024):
        """[Phase 2] 执行投影 (LSH)"""
        if not self.initialized: self.initialize_enclave()

        total_len = w_new_flat.size
        if ranges is None:
            ranges_flat = np.array([0, total_len], dtype=np.int32)
        else:
            ranges_flat = np.array(ranges, dtype=np.int32).flatten()
            
        output = np.zeros(output_dim, dtype=np.float32)
        
        ret = self.lib.tee_secure_aggregation_phase(
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
            # 尝试重试一次 (针对多进程 Fork 后 Enclave 失效问题)
            self.initialized = False
            self.initialize_enclave()
            ret = self.lib.tee_secure_aggregation_phase(
                seed, w_new_flat, w_old_flat, total_len, ranges_flat, len(ranges_flat), output, output_dim
            )
            if ret != 0:
                raise RuntimeError("TEE secure projection failed after retry!")
            
        return output

    def generate_masked_gradient(self, seed_mask_root, seed_global_0, client_id, k_weight, n_ratio, w_new_flat, w_old_flat, ranges):
        """[Phase 4] 生成双掩码加密梯度"""
        if not self.initialized: self.initialize_enclave()
        
        total_len = w_new_flat.size
        if ranges is None:
            ranges_flat = np.array([0, total_len], dtype=np.int32)
        else:
            ranges_flat = np.array(ranges, dtype=np.int32).flatten()
            
        # 准备输出 buffer
        # 虽然类型是 float32，但 Enclave 内部写入的是定点化后的 int32 数据
        output_buffer = np.zeros(total_len, dtype=np.float32)
        
        ret = self._func_generate_masked(
            seed_mask_root,
            seed_global_0,
            client_id,
            k_weight,
            n_ratio,
            w_new_flat,
            w_old_flat,
            total_len,
            ranges_flat,
            len(ranges_flat),
            output_buffer,
            total_len
        )
        if ret != 0:
             # 尝试重试
            self.initialized = False
            self.initialize_enclave()
            ret = self._func_generate_masked(
                seed_mask_root, seed_global_0, client_id, k_weight, n_ratio,
                w_new_flat, w_old_flat, total_len, ranges_flat, len(ranges_flat), output_buffer, total_len
            )
            if ret != 0:
                raise RuntimeError("TEE generate masked gradient failed!")
        
        # [关键] 将 float32 的内存重新解释为 int32
        # 这是为了配合 TEE 内部的定点化整数运算，Python 侧拿到的是整数数组
        encrypted_int = output_buffer.view(np.int32)
        
        return encrypted_int

    def get_vector_shares(self, seed_sss, seed_mask_root, target_client_id, threshold, total_clients):
        """[Phase 5] 批量获取恢复分片 (Alpha 和 Beta)"""
        if not self.initialized: self.initialize_enclave()
        
        # 分配结构体数组
        ShareArrayType = SharePackage * total_clients
        output_shares = ShareArrayType()
        
        ret = self._func_get_shares(
            seed_sss,
            seed_mask_root,
            target_client_id,
            threshold,
            total_clients,
            output_shares
        )
        if ret != 0:
            self.initialized = False
            self.initialize_enclave()
            ret = self._func_get_shares(
                seed_sss, seed_mask_root, target_client_id, threshold, total_clients, output_shares
            )
            if ret != 0:
                raise RuntimeError("TEE vector shares generation failed!")
        
        # 转换为 Python 友好的格式 (List of Dicts)
        # 注意: 结构体中的 float 实际上存储的是 int64 转换后的值
        res = []
        for i in range(total_clients):
            res.append({
                'alpha': int(output_shares[i].share_alpha),
                'beta': int(output_shares[i].share_beta)
            })
        return res

    def close(self):
        if self.lib:
            try:
                self.lib.tee_destroy()
            except:
                pass
        self.initialized = False