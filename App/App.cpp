/* App/App.cpp */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sgx_urts.h"
#include "App.h"
#include "Enclave_u.h"

sgx_enclave_id_t global_eid = 0;

void print_error_message(sgx_status_t ret) {
    printf("[Bridge Error] SGX Status: 0x%X\n", ret);
}

void ocall_print_string(const char *str) {
    printf("%s", str);
}

extern "C" {

int tee_init(const char* enclave_path) {
    if (global_eid != 0) return 0;
    sgx_status_t ret = sgx_create_enclave(enclave_path, SGX_DEBUG_FLAG, NULL, NULL, &global_eid, NULL);
    if (ret != SGX_SUCCESS) {
        print_error_message(ret);
        return -1;
    }
    // printf("[Bridge] App: Enclave created successfully (EID: %lu).\n", global_eid);
    return 0;
}

void tee_destroy() {
    if (global_eid != 0) {
        sgx_destroy_enclave(global_eid);
        global_eid = 0;
    }
}

void tee_prepare_gradient(
    int client_id, 
    const char* proj_seed_str, 
    float* w_new, float* w_old, size_t model_len, 
    int* ranges, size_t ranges_len, float* output_proj, size_t out_len
) {
    sgx_status_t ret = ecall_prepare_gradient(
        global_eid, client_id, proj_seed_str, 
        w_new, w_old, model_len, ranges, ranges_len, output_proj, out_len
    );
    if (ret != SGX_SUCCESS) print_error_message(ret);
}

void tee_generate_masked_gradient_dynamic(
    const char* seed_mask_root_str, 
    const char* seed_global_0_str, 
    int client_id, 
    int* active_ids, size_t active_count,
    const char* k_weight_str, 
    size_t model_len, 
    int* ranges, size_t ranges_len, 
    long long* output, size_t out_len
) {
    sgx_status_t ret = ecall_generate_masked_gradient_dynamic(
        global_eid, seed_mask_root_str, seed_global_0_str, client_id, 
        active_ids, active_count, k_weight_str, 
        model_len, ranges, ranges_len, output, out_len
    );
    if (ret != SGX_SUCCESS) print_error_message(ret);
}

void tee_get_vector_shares_dynamic(
    const char* seed_sss_str, 
    const char* seed_mask_root_str, 
    int* u1_ids, size_t u1_len, 
    int* u2_ids, size_t u2_len, 
    int my_client_id, 
    int threshold, 
    long long* output_vector, 
    size_t out_max_len
) {
    sgx_status_t ret = ecall_get_vector_shares_dynamic(
        global_eid, seed_sss_str, seed_mask_root_str, 
        u1_ids, u1_len, u2_ids, u2_len, my_client_id, threshold, 
        output_vector, out_max_len
    );
    if (ret != SGX_SUCCESS) print_error_message(ret);
}

void tee_generate_noise_from_seed(
    const char* seed_str, 
    size_t len, 
    long long* output
) {
    sgx_status_t ret = ecall_generate_noise_from_seed(
        global_eid, seed_str, len, output
    );
    if (ret != SGX_SUCCESS) print_error_message(ret);
}

} // extern "C"