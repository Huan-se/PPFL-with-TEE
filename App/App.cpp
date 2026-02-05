/* App/App.cpp */
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <pwd.h>

#include "sgx_urts.h"
#include "App.h"
#include "Enclave_u.h" // 自动生成的头文件，包含 ecall 定义

/* 全局 Enclave ID */
sgx_enclave_id_t global_eid = 0;

/* OCall 实现: Enclave 内部调用打印函数 */
extern "C" void ocall_print_string(const char *str) {
    printf("%s", str);
}

/* 初始化 Enclave (Python只调用一次) */
extern "C" int tee_init(const char* enclave_filename) {
    sgx_status_t ret = SGX_ERROR_UNEXPECTED;
    
    // 如果已初始化，直接返回成功 (避免多线程竞争报错)
    if (global_eid != 0) return SGX_SUCCESS;

    ret = sgx_create_enclave(enclave_filename, SGX_DEBUG_FLAG, NULL, NULL, &global_eid, NULL);
    
    if (ret != SGX_SUCCESS) {
        printf("[Bridge] App: error %#x, failed to create enclave.\n", ret);
        return -1;
    }
    printf("[Bridge] App: Enclave created successfully (EID: %lu).\n", global_eid);
    return 0;
}

/* 销毁 Enclave */
extern "C" void tee_destroy() {
    if (global_eid != 0) {
        sgx_destroy_enclave(global_eid);
        global_eid = 0;
    }
}

/* * [Phase 2] Prepare Gradient Bridge
 * Python: tee_prepare_gradient(...)
 * -> C++: ecall_prepare_gradient(...)
 */
extern "C" void tee_prepare_gradient(
    int client_id,
    long proj_seed, 
    float* w_new, 
    float* w_old, 
    int model_len, 
    int* ranges, 
    int ranges_len, 
    float* output_proj, 
    int out_len
) {
    if (global_eid == 0) { printf("[Bridge] Error: Enclave not initialized\n"); return; }
    
    sgx_status_t status;
    // 调用 EDL 生成的代理函数
    // 注意: size_t 在 Python ctypes 传参时需确保一致，这里用 int 接收后强转也可，只要不超过范围
    status = ecall_prepare_gradient(
        global_eid, 
        client_id, 
        proj_seed, 
        w_new, 
        w_old, 
        (size_t)model_len, 
        ranges, 
        (size_t)ranges_len, 
        output_proj, 
        (size_t)out_len
    );

    if (status != SGX_SUCCESS) {
        printf("[Bridge] Ecall prepare_gradient failed: %#x\n", status);
    }
}

/* * [Phase 4] Generate Masked Gradient Bridge (Stateful)
 * Python: tee_generate_masked_gradient_dynamic(...)
 * -> C++: ecall_generate_masked_gradient_dynamic(...)
 */
extern "C" void tee_generate_masked_gradient_dynamic(
    long seed_mask_root, 
    long seed_global_0, 
    int client_id, 
    float k_weight, 
    float n_ratio, 
    int model_len, 
    int* ranges, 
    int ranges_len, 
    long long* output, 
    int out_len
) {
    if (global_eid == 0) { printf("[Bridge] Error: Enclave not initialized\n"); return; }

    sgx_status_t status;
    status = ecall_generate_masked_gradient_dynamic(
        global_eid,
        seed_mask_root,
        seed_global_0,
        client_id,
        k_weight,
        n_ratio,
        (size_t)model_len,
        ranges,
        (size_t)ranges_len,
        output,
        (size_t)out_len
    );

    if (status != SGX_SUCCESS) {
        printf("[Bridge] Ecall generate_masked failed: %#x\n", status);
    }
}

/* * [Phase 5] Get Shares Bridge
 */
extern "C" void tee_get_vector_shares_dynamic(
    long seed_sss, 
    long seed_mask_root, 
    int target_client_id, 
    int threshold, 
    int total_clients, 
    void* output_shares // void* to struct buffer
) {
    if (global_eid == 0) { printf("[Bridge] Error: Enclave not initialized\n"); return; }

    sgx_status_t status;
    // 强制转换 output_shares 为 EDL 中定义的 struct 指针类型
    // 在 C++ 侧这是 struct SharePackage*，由 Enclave_u.h 定义
    status = ecall_get_vector_shares_dynamic(
        global_eid,
        seed_sss,
        seed_mask_root,
        target_client_id,
        threshold,
        total_clients,
        (struct SharePackage*)output_shares
    );

    if (status != SGX_SUCCESS) {
        printf("[Bridge] Ecall get_shares failed: %#x\n", status);
    }
}