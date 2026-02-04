# -------- 配置开始 --------

# SGX SDK 路径
SGX_SDK ?= /opt/intel/sgxsdk
# 工具路径
SGX_EDGER8R := $(SGX_SDK)/bin/x64/sgx_edger8r
SGX_ENCLAVE_SIGNER := $(SGX_SDK)/bin/x64/sgx_sign

SGX_MODE ?= HW
SGX_ARCH ?= x64
SGX_DEBUG ?= 1

# Eigen 路径
EIGEN_PATH := Enclave/include/eigen-3.3.9
# GCC 内置头文件路径
GCC_BUILTIN_INC := $(shell $(CC) -print-file-name=include)
# [新增] 输出目录
OUT_DIR := lib

include $(SGX_SDK)/buildenv.mk

# === 库名称定义 ===
ifeq ($(SGX_MODE), HW)
	Urts_Library_Name := sgx_urts
	Trts_Library_Name := sgx_trts
	Service_Library_Name := sgx_uae_service
	TService_Library_Name := sgx_tservice
else
	Urts_Library_Name := sgx_urts_sim
	Trts_Library_Name := sgx_trts_sim
	Service_Library_Name := sgx_uae_service_sim
	TService_Library_Name := sgx_tservice_sim
endif

# App (不可信部分) 配置
App_Cpp_Files := App/App.cpp
App_Include_Paths := -IApp -I$(SGX_SDK)/include
App_C_Flags := -fPIC -Wno-attributes $(App_Include_Paths)
App_Link_Flags := -L$(SGX_SDK)/lib64 -l$(Urts_Library_Name) -l$(Service_Library_Name) -lpthread -shared

# Enclave (可信部分) 配置
Enclave_Cpp_Files := Enclave/Enclave.cpp
Enclave_Include_Paths := -IEnclave -I$(SGX_SDK)/include -I$(SGX_SDK)/include/tlibc -I$(SGX_SDK)/include/libcxx -I$(EIGEN_PATH) -I$(GCC_BUILTIN_INC)

Enclave_C_Flags := -nostdinc -fvisibility=hidden -fpie -fstack-protector $(Enclave_Include_Paths)
Enclave_Cpp_Flags := $(Enclave_C_Flags) -std=c++11 -nostdinc++

Enclave_Link_Flags := -Wl,--no-undefined -nostdlib -nodefaultlibs -nostartfiles -L$(SGX_SDK)/lib64 \
	-Wl,--whole-archive -l$(Trts_Library_Name) -Wl,--no-whole-archive \
	-Wl,--start-group -lsgx_tstdc -lsgx_tcxx -l$(TService_Library_Name) -lsgx_tcrypto -Wl,--end-group \
	-Wl,-Bstatic -Wl,-Bsymbolic -Wl,--no-undefined \
	-Wl,-pie,-eenclave_entry -Wl,--export-dynamic \
	-Wl,--defsym,__ImageBase=0

# -------- 目标规则 --------

.PHONY: all clean directories

# 默认目标：先创建目录，再编译库
all: directories $(OUT_DIR)/libtee_bridge.so $(OUT_DIR)/enclave.signed.so

# 创建输出目录
directories:
	@mkdir -p $(OUT_DIR)

# 1. 生成胶水代码 (保持在原处，方便调试)
App/Enclave_u.c App/Enclave_u.h: $(SGX_EDGER8R) Enclave/Enclave.edl
	cd App && $(SGX_EDGER8R) --untrusted ../Enclave/Enclave.edl --search-path ../Enclave --search-path $(SGX_SDK)/include

Enclave/Enclave_t.c Enclave/Enclave_t.h: $(SGX_EDGER8R) Enclave/Enclave.edl
	cd Enclave && $(SGX_EDGER8R) --trusted ../Enclave/Enclave.edl --search-path ../Enclave --search-path $(SGX_SDK)/include

# 2. 编译 App -> 输出到 lib/libtee_bridge.so
App/Enclave_u.o: App/Enclave_u.c
	$(CC) $(App_C_Flags) -c $< -o $@

App/App.o: App/App.cpp App/Enclave_u.h
	$(CXX) $(App_C_Flags) -std=c++11 -c $< -o $@

$(OUT_DIR)/libtee_bridge.so: App/Enclave_u.o App/App.o
	$(CXX) $^ -o $@ $(App_Link_Flags)
	@echo "LINK =>  $@"

# 3. 编译 Enclave -> 输出到 lib/enclave.signed.so
Enclave/Enclave_t.o: Enclave/Enclave_t.c
	$(CC) $(Enclave_C_Flags) -c $< -o $@

Enclave/Enclave.o: Enclave/Enclave.cpp
	$(CXX) $(Enclave_Cpp_Flags) -c $< -o $@

# 中间产物 enclave.so 也放进 lib
$(OUT_DIR)/enclave.so: Enclave/Enclave_t.o Enclave/Enclave.o
	$(CXX) $^ -o $@ $(Enclave_Link_Flags)

$(OUT_DIR)/enclave.signed.so: $(OUT_DIR)/enclave.so
	$(SGX_ENCLAVE_SIGNER) sign -key Enclave/Enclave_private_test.pem -enclave $(OUT_DIR)/enclave.so -out $@ -config Enclave/Enclave.config.xml
	@echo "SIGN =>  $@"

clean:
	rm -rf $(OUT_DIR)
	rm -f App/*.o App/Enclave_u.*
	rm -f Enclave/*.o Enclave/Enclave_t.*