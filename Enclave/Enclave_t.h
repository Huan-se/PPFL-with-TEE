#ifndef ENCLAVE_T_H__
#define ENCLAVE_T_H__

#include <stdint.h>
#include <wchar.h>
#include <stddef.h>
#include "sgx_edger8r.h" /* for sgx_ocall etc. */


#include <stdlib.h> /* for size_t */

#define SGX_CAST(type, item) ((type)(item))

#ifdef __cplusplus
extern "C" {
#endif

void ecall_set_verbose(int level);
void ecall_prepare_gradient(int client_id, const char* proj_seed_str, float* w_new, float* w_old, size_t model_len, int* ranges, size_t ranges_len, float* output_proj, size_t out_len);
void ecall_generate_masked_gradient_dynamic(const char* seed_mask_root_str, const char* seed_global_0_str, int client_id, int* active_ids, size_t active_count, const char* k_weight_str, size_t model_len, int* ranges, size_t ranges_len, long long* output, size_t out_len);
void ecall_get_vector_shares_dynamic(const char* seed_sss_str, const char* seed_mask_root_str, int* u1_ids, size_t u1_len, int* u2_ids, size_t u2_len, int my_client_id, int threshold, long long* output_vector, size_t out_max_len);
void ecall_generate_noise_from_seed(const char* seed_str, size_t len, long long* output);

sgx_status_t SGX_CDECL ocall_print_string(const char* str);
sgx_status_t SGX_CDECL sgx_oc_cpuidex(int cpuinfo[4], int leaf, int subleaf);
sgx_status_t SGX_CDECL sgx_thread_wait_untrusted_event_ocall(int* retval, const void* self);
sgx_status_t SGX_CDECL sgx_thread_set_untrusted_event_ocall(int* retval, const void* waiter);
sgx_status_t SGX_CDECL sgx_thread_setwait_untrusted_events_ocall(int* retval, const void* waiter, const void* self);
sgx_status_t SGX_CDECL sgx_thread_set_multiple_untrusted_events_ocall(int* retval, const void** waiters, size_t total);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
