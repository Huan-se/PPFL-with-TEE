import ctypes
import os
import numpy as np

# 定义与 Enclave 接口一致的结构体
class SharePackage(ctypes.Structure):
    _fields_ = [
        ("share_alpha", ctypes.c_double),
        ("share_beta", ctypes.c_double)
    ]

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
        
        # =========================================================
        # [Phase 2] 准备梯度 (Stateful Step 1)
        # 对应 App.cpp: tee_prepare_gradient
        # =========================================================
        try:
            func_prep = self.lib.tee_prepare_gradient
        except AttributeError:
            # 兼容可能的命名差异
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
        # [Phase 4] 生成双掩码梯度 (Stateful Step 2)
        # 对应 App.cpp: tee_generate_masked_gradient_dynamic
        # 关键修改：不再传入 w_new/w_old
        # =========================================================
        try:
            func_mask = self.lib.tee_generate_masked_gradient_dynamic
        except AttributeError:
            func_mask = self.lib.ecall_generate_masked_gradient_dynamic
            
        func_mask.argtypes = [
            ctypes.c_long,  # seed_mask_root
            ctypes.c_long,  # seed_global_0
            ctypes.c_int,   # client_id
            ctypes.c_float, # k_weight
            ctypes.c_float, # n_ratio
            ctypes.c_int,   # model_len (仅用于 bounds check 和循环)
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),   # ranges
            ctypes.c_int,   # ranges_len
            np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS'),   # output (int64 buffer)
            ctypes.c_int    # out_len
        ]
        func_mask.restype = None
        self._func_generate = func_mask

        # =========================================================
        # [Phase 5] 向量化秘密共享
        # 对应 App.cpp: tee_get_vector_shares_dynamic
        # =========================================================
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
        func_sss.restype = None
        self._func_get_shares = func_sss

    def initialize_enclave(self, enclave_path=None):
        """初始化 Enclave (全局单例模式)"""
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

        # 调用 tee_init
        # 注意：SGX Enclave 在一个进程内只能初始化一次
        # 如果返回非0，可能是 "SGX_ERROR_ENCLAVE_ALREADY_INITIALIZED"，我们忽略它
        res = self.lib.tee_init(enclave_path.encode('utf-8'))
        
        self.initialized = True

    # =========================================================
    # Python 调用接口
    # =========================================================

    def prepare_gradient(self, client_id, proj_seed, w_new, w_old, output_dim=1024):
        """
        [Phase 2] 将模型传入 TEE，计算梯度并锁定状态，返回 LSH 投影
        """
        if not self.initialized: self.initialize_enclave()
        
        total_len = w_new.size
        # 默认处理整个模型
        ranges = np.array([0, total_len], dtype=np.int32)
        
        # 分配输出 buffer
        output_proj = np.zeros(output_dim, dtype=np.float32)
        
        self._func_prepare(
            client_id, 
            proj_seed, 
            w_new, 
            w_old, 
            total_len, 
            ranges, 
            len(ranges), 
            output_proj, 
            output_dim
        )
        return output_proj

    def generate_masked_gradient(self, seed_mask, seed_g0, cid, k, n, model_len=0, ranges=None):
        """
        [Phase 4] 读取 TEE 内部锁定的梯度，加权加噪，返回密文
        Args:
            model_len: 必须提供，用于分配 Python 侧的接收 buffer
        """
        if not self.initialized: self.initialize_enclave()
        
        if ranges is None: 
            ranges = np.array([0, model_len], dtype=np.int32)
        
        # 分配输出 buffer (int64)
        out_buf = np.zeros(model_len, dtype=np.int64)
        
        # 注意：这里不再传入 w_new / w_old
        self._func_generate(
            seed_mask, 
            seed_g0, 
            cid, 
            k, 
            n, 
            model_len, 
            ranges, 
            len(ranges), 
            out_buf, 
            model_len
        )
        return out_buf

    def get_vector_shares(self, seed_sss, seed_mask, target, threshold, total):
        """
        [Phase 5] 恢复密钥分片
        """
        if not self.initialized: self.initialize_enclave()
        
        # 分配结构体数组
        out = (SharePackage * total)()
        
        self._func_get_shares(
            seed_sss, 
            seed_mask, 
            target, 
            threshold, 
            total, 
            out
        )
        
        # 转换为 Python 字典列表返回
        return [{'alpha': int(o.share_alpha), 'beta': int(o.share_beta)} for o in out]