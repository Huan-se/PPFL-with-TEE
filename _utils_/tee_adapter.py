import ctypes
import os
import numpy as np

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
        """加载 .so 库并配置参数类型 (可在反序列化时复用)"""
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
        
        # 3. 核心安全聚合接口
        self.lib.tee_secure_aggregation.argtypes = [
            ctypes.c_long,
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
            ctypes.c_int
        ]
        self.lib.tee_secure_aggregation.restype = ctypes.c_int

    def __getstate__(self):
        """序列化前调用：移除不可 pickle 的 ctypes 对象"""
        state = self.__dict__.copy()
        if 'lib' in state:
            del state['lib'] 
        # 关键：子进程收到对象时，Enclave 肯定还没在那个进程初始化，所以强制设为 False
        state['initialized'] = False 
        return state

    def __setstate__(self, state):
        """反序列化后调用：重新加载动态库"""
        self.__dict__.update(state)
        self._load_library() # 在新进程中重新加载 .so

    def initialize_enclave(self, enclave_path=None):
        """初始化 Enclave"""
        if self.initialized:
            return # 当前进程已初始化，跳过

        if enclave_path is None:
            # 尝试使用之前保存的路径，或者是默认路径
            if self.enclave_path:
                enclave_path = self.enclave_path
            else:
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                enclave_path = os.path.join(base_dir, "lib", "enclave.signed.so")
            
        if not os.path.exists(enclave_path):
            raise FileNotFoundError(f"Signed Enclave not found at: {enclave_path}")

        # 记录路径供后续（子进程）使用
        self.enclave_path = enclave_path

        # 调用 C 接口
        ret = self.lib.tee_init(enclave_path.encode('utf-8'))
        if ret != 0:
            raise RuntimeError(f"Failed to initialize SGX Enclave (PID: {os.getpid()})")
            
        self.initialized = True
        # print(f"[TEEAdapter] Enclave initialized in PID {os.getpid()}.")

    def secure_project(self, seed, w_new_flat, w_old_flat, ranges, output_dim=1024):
        """执行投影"""
        # [关键]：如果是子进程第一次调用，这里会自动初始化 Enclave
        if not self.initialized:
            self.initialize_enclave()

        # 数据检查
        if w_new_flat.shape != w_old_flat.shape:
            raise ValueError("Shape mismatch between w_new and w_old")
        
        total_len = w_new_flat.size
        
        if not ranges:
            ranges_flat = np.array([0, total_len], dtype=np.int32)
        else:
            ranges_flat = np.array(ranges, dtype=np.int32).flatten()
            
        output = np.zeros(output_dim, dtype=np.float32)
        
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
            # 如果失败，可能是 Enclave 崩溃或 ID 失效，尝试重置一次
            print(f"[TEEAdapter] Warning: secure aggregation failed in PID {os.getpid()}, retrying init...")
            self.initialized = False
            self.initialize_enclave()
            ret = self.lib.tee_secure_aggregation(
                seed, w_new_flat, w_old_flat, total_len, ranges_flat, len(ranges_flat), output, output_dim
            )
            if ret != 0:
                raise RuntimeError("TEE secure aggregation failed after retry!")
            
        return output

    def close(self):
        if self.lib:
            try:
                self.lib.tee_destroy()
            except:
                pass
        self.initialized = False