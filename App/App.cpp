/* App/App.cpp */
#include <stdio.h>
#include <sgx_urts.h>
#include "App.h"
#include "Enclave_u.h"

sgx_enclave_id_t global_eid = 0;

void ocall_print_string(const char *str) {
    printf("%s", str);
}

extern "C" {
    int tee_init(const char* enclave_filename) {
        sgx_status_t ret = sgx_create_enclave(enclave_filename, SGX_DEBUG_FLAG, NULL, NULL, &global_eid, NULL);
        if (ret != SGX_SUCCESS) { printf("Error: 0x%x\n", ret); return -1; }
        return 0;
    }

    void tee_destroy() {
        if (global_eid != 0) { sgx_destroy_enclave(global_eid); global_eid = 0; }
    }

    // 1. 投影
    int tee_secure_aggregation(long seed, float* w_new, float* w_old, int model_len, int* ranges, int ranges_len, float* output, int out_len) {
        if (global_eid == 0) return -1;
        sgx_status_t status = ecall_secure_aggregation_phase(
            global_eid, seed, w_new, w_old, (size_t)model_len, ranges, (size_t)ranges_len, output, (size_t)out_len
        );
        if (status != SGX_SUCCESS) { printf("Ecall Projection Failed: 0x%x\n", status); return -1; }
        return 0;
    }

    // 2. 双掩码
    int tee_generate_masked_gradient(
        long seed_r, long seed_b, float weight,
        float* w_new, float* w_old, int model_len,
        int* ranges, int ranges_len,
        float* output, int out_len
    ) {
        if (global_eid == 0) return -1;
        sgx_status_t status = ecall_generate_masked_gradient(
            global_eid, seed_r, seed_b, weight,
            w_new, w_old, (size_t)model_len,
            ranges, (size_t)ranges_len,
            output, (size_t)out_len
        );
        if (status != SGX_SUCCESS) { printf("Ecall Masked Grad Failed: 0x%x\n", status); return -1; }
        return 0;
    }

    // 3. 恢复
    int tee_get_recovery_share(long seed_sss, float secret_val, int threshold, int target_x, float* share_val) {
        if (global_eid == 0) return -1;
        sgx_status_t status = ecall_get_recovery_share(
            global_eid, seed_sss, secret_val, threshold, target_x, share_val
        );
        if (status != SGX_SUCCESS) { printf("Ecall SSS Failed: 0x%x\n", status); return -1; }
        return 0;
    }
}