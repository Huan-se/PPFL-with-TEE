/* App/App.cpp */
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <pwd.h>

#include "sgx_urts.h"
#include "App.h"
#include "Enclave_u.h" // 自动生成的头文件，包含更新后的 ecall 定义

/* 全局 Enclave ID */
sgx_enclave_id_t global_eid = 0;

/* OCall 实现: Enclave 内部调用打印函数 */
extern "C" void ocall_print_string(const char *str) {
    printf("%s", str);
}

/* 初始化 Enclave */
extern "C" int tee_init(const char* enclave_filename) {
    sgx_status_t ret = SGX_ERROR_UNEXPECTED;
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

/* [Phase 2] Prepare Gradient Bridge */
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

/* [Phase 4] Generate Masked Gradient Bridge (Updated for V2) */
extern "C" void tee_generate_masked_gradient_dynamic(
    long seed_mask_root, 
    long seed_global_0, 
    int client_id, 
    int* active_ids,    // [新增]
    int active_len,     // [新增]
    float k_weight, 
    int model_len, 
    int* ranges, 
    int ranges_len, 
    long long* output, 
    int out_len
) {
    if (global_eid == 0) { printf("[Bridge] Error: Enclave not initialized\n"); return; }

    sgx_status_t status;
    // 这里的参数顺序必须严格匹配 Enclave_u.h 中的定义 (由 EDL 生成)
    status = ecall_generate_masked_gradient_dynamic(
        global_eid,
        seed_mask_root,
        seed_global_0,
        client_id,
        active_ids,         // 传入 int*
        (size_t)active_len, // 传入 size_t
        k_weight,
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

/* [Phase 5] Get Shares Bridge (Updated for V2) */
extern "C" void tee_get_vector_shares_dynamic(
    long seed_sss, 
    long seed_mask_root, 
    int* u1_ids,        // [新增]
    int u1_len,         // [新增]
    int* u2_ids,        // [新增]
    int u2_len,         // [新增]
    int my_client_id, 
    int threshold, 
    long long* output_vector, // [变更] 输出类型改为 long long*
    int out_max_len
) {
    if (global_eid == 0) { printf("[Bridge] Error: Enclave not initialized\n"); return; }

    sgx_status_t status;
    status = ecall_get_vector_shares_dynamic(
        global_eid,
        seed_sss,
        seed_mask_root,
        u1_ids, (size_t)u1_len,
        u2_ids, (size_t)u2_len,
        my_client_id,
        threshold,
        output_vector,
        (size_t)out_max_len
    );

    if (status != SGX_SUCCESS) {
        printf("[Bridge] Ecall get_shares failed: %#x\n", status);
    }
}

/* [Server Helper] Generate Noise Bridge (New) */
extern "C" void tee_generate_noise_from_seed(
    long seed, 
    int len, 
    long long* output
) {
    if (global_eid == 0) { printf("[Bridge] Error: Enclave not initialized\n"); return; }

    sgx_status_t status;
    status = ecall_generate_noise_from_seed(
        global_eid,
        seed,
        (size_t)len,
        output
    );
    
    if (status != SGX_SUCCESS) {
        printf("[Bridge] Ecall generate_noise failed: %#x\n", status);
    }
}