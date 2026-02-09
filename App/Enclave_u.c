#include "Enclave_u.h"
#include <errno.h>

typedef struct ms_ecall_set_verbose_t {
	int ms_level;
} ms_ecall_set_verbose_t;

typedef struct ms_ecall_prepare_gradient_t {
	int ms_client_id;
	const char* ms_proj_seed_str;
	size_t ms_proj_seed_str_len;
	float* ms_w_new;
	float* ms_w_old;
	size_t ms_model_len;
	int* ms_ranges;
	size_t ms_ranges_len;
	float* ms_output_proj;
	size_t ms_out_len;
} ms_ecall_prepare_gradient_t;

typedef struct ms_ecall_generate_masked_gradient_dynamic_t {
	const char* ms_seed_mask_root_str;
	size_t ms_seed_mask_root_str_len;
	const char* ms_seed_global_0_str;
	size_t ms_seed_global_0_str_len;
	int ms_client_id;
	int* ms_active_ids;
	size_t ms_active_count;
	const char* ms_k_weight_str;
	size_t ms_k_weight_str_len;
	size_t ms_model_len;
	int* ms_ranges;
	size_t ms_ranges_len;
	long long* ms_output;
	size_t ms_out_len;
} ms_ecall_generate_masked_gradient_dynamic_t;

typedef struct ms_ecall_get_vector_shares_dynamic_t {
	const char* ms_seed_sss_str;
	size_t ms_seed_sss_str_len;
	const char* ms_seed_mask_root_str;
	size_t ms_seed_mask_root_str_len;
	int* ms_u1_ids;
	size_t ms_u1_len;
	int* ms_u2_ids;
	size_t ms_u2_len;
	int ms_my_client_id;
	int ms_threshold;
	long long* ms_output_vector;
	size_t ms_out_max_len;
} ms_ecall_get_vector_shares_dynamic_t;

typedef struct ms_ecall_generate_noise_from_seed_t {
	const char* ms_seed_str;
	size_t ms_seed_str_len;
	size_t ms_len;
	long long* ms_output;
} ms_ecall_generate_noise_from_seed_t;

typedef struct ms_ocall_print_string_t {
	const char* ms_str;
} ms_ocall_print_string_t;

typedef struct ms_sgx_oc_cpuidex_t {
	int* ms_cpuinfo;
	int ms_leaf;
	int ms_subleaf;
} ms_sgx_oc_cpuidex_t;

typedef struct ms_sgx_thread_wait_untrusted_event_ocall_t {
	int ms_retval;
	const void* ms_self;
} ms_sgx_thread_wait_untrusted_event_ocall_t;

typedef struct ms_sgx_thread_set_untrusted_event_ocall_t {
	int ms_retval;
	const void* ms_waiter;
} ms_sgx_thread_set_untrusted_event_ocall_t;

typedef struct ms_sgx_thread_setwait_untrusted_events_ocall_t {
	int ms_retval;
	const void* ms_waiter;
	const void* ms_self;
} ms_sgx_thread_setwait_untrusted_events_ocall_t;

typedef struct ms_sgx_thread_set_multiple_untrusted_events_ocall_t {
	int ms_retval;
	const void** ms_waiters;
	size_t ms_total;
} ms_sgx_thread_set_multiple_untrusted_events_ocall_t;

static sgx_status_t SGX_CDECL Enclave_ocall_print_string(void* pms)
{
	ms_ocall_print_string_t* ms = SGX_CAST(ms_ocall_print_string_t*, pms);
	ocall_print_string(ms->ms_str);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL Enclave_sgx_oc_cpuidex(void* pms)
{
	ms_sgx_oc_cpuidex_t* ms = SGX_CAST(ms_sgx_oc_cpuidex_t*, pms);
	sgx_oc_cpuidex(ms->ms_cpuinfo, ms->ms_leaf, ms->ms_subleaf);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL Enclave_sgx_thread_wait_untrusted_event_ocall(void* pms)
{
	ms_sgx_thread_wait_untrusted_event_ocall_t* ms = SGX_CAST(ms_sgx_thread_wait_untrusted_event_ocall_t*, pms);
	ms->ms_retval = sgx_thread_wait_untrusted_event_ocall(ms->ms_self);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL Enclave_sgx_thread_set_untrusted_event_ocall(void* pms)
{
	ms_sgx_thread_set_untrusted_event_ocall_t* ms = SGX_CAST(ms_sgx_thread_set_untrusted_event_ocall_t*, pms);
	ms->ms_retval = sgx_thread_set_untrusted_event_ocall(ms->ms_waiter);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL Enclave_sgx_thread_setwait_untrusted_events_ocall(void* pms)
{
	ms_sgx_thread_setwait_untrusted_events_ocall_t* ms = SGX_CAST(ms_sgx_thread_setwait_untrusted_events_ocall_t*, pms);
	ms->ms_retval = sgx_thread_setwait_untrusted_events_ocall(ms->ms_waiter, ms->ms_self);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL Enclave_sgx_thread_set_multiple_untrusted_events_ocall(void* pms)
{
	ms_sgx_thread_set_multiple_untrusted_events_ocall_t* ms = SGX_CAST(ms_sgx_thread_set_multiple_untrusted_events_ocall_t*, pms);
	ms->ms_retval = sgx_thread_set_multiple_untrusted_events_ocall(ms->ms_waiters, ms->ms_total);

	return SGX_SUCCESS;
}

static const struct {
	size_t nr_ocall;
	void * table[6];
} ocall_table_Enclave = {
	6,
	{
		(void*)Enclave_ocall_print_string,
		(void*)Enclave_sgx_oc_cpuidex,
		(void*)Enclave_sgx_thread_wait_untrusted_event_ocall,
		(void*)Enclave_sgx_thread_set_untrusted_event_ocall,
		(void*)Enclave_sgx_thread_setwait_untrusted_events_ocall,
		(void*)Enclave_sgx_thread_set_multiple_untrusted_events_ocall,
	}
};
sgx_status_t ecall_set_verbose(sgx_enclave_id_t eid, int level)
{
	sgx_status_t status;
	ms_ecall_set_verbose_t ms;
	ms.ms_level = level;
	status = sgx_ecall(eid, 0, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_prepare_gradient(sgx_enclave_id_t eid, int client_id, const char* proj_seed_str, float* w_new, float* w_old, size_t model_len, int* ranges, size_t ranges_len, float* output_proj, size_t out_len)
{
	sgx_status_t status;
	ms_ecall_prepare_gradient_t ms;
	ms.ms_client_id = client_id;
	ms.ms_proj_seed_str = proj_seed_str;
	ms.ms_proj_seed_str_len = proj_seed_str ? strlen(proj_seed_str) + 1 : 0;
	ms.ms_w_new = w_new;
	ms.ms_w_old = w_old;
	ms.ms_model_len = model_len;
	ms.ms_ranges = ranges;
	ms.ms_ranges_len = ranges_len;
	ms.ms_output_proj = output_proj;
	ms.ms_out_len = out_len;
	status = sgx_ecall(eid, 1, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_generate_masked_gradient_dynamic(sgx_enclave_id_t eid, const char* seed_mask_root_str, const char* seed_global_0_str, int client_id, int* active_ids, size_t active_count, const char* k_weight_str, size_t model_len, int* ranges, size_t ranges_len, long long* output, size_t out_len)
{
	sgx_status_t status;
	ms_ecall_generate_masked_gradient_dynamic_t ms;
	ms.ms_seed_mask_root_str = seed_mask_root_str;
	ms.ms_seed_mask_root_str_len = seed_mask_root_str ? strlen(seed_mask_root_str) + 1 : 0;
	ms.ms_seed_global_0_str = seed_global_0_str;
	ms.ms_seed_global_0_str_len = seed_global_0_str ? strlen(seed_global_0_str) + 1 : 0;
	ms.ms_client_id = client_id;
	ms.ms_active_ids = active_ids;
	ms.ms_active_count = active_count;
	ms.ms_k_weight_str = k_weight_str;
	ms.ms_k_weight_str_len = k_weight_str ? strlen(k_weight_str) + 1 : 0;
	ms.ms_model_len = model_len;
	ms.ms_ranges = ranges;
	ms.ms_ranges_len = ranges_len;
	ms.ms_output = output;
	ms.ms_out_len = out_len;
	status = sgx_ecall(eid, 2, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_get_vector_shares_dynamic(sgx_enclave_id_t eid, const char* seed_sss_str, const char* seed_mask_root_str, int* u1_ids, size_t u1_len, int* u2_ids, size_t u2_len, int my_client_id, int threshold, long long* output_vector, size_t out_max_len)
{
	sgx_status_t status;
	ms_ecall_get_vector_shares_dynamic_t ms;
	ms.ms_seed_sss_str = seed_sss_str;
	ms.ms_seed_sss_str_len = seed_sss_str ? strlen(seed_sss_str) + 1 : 0;
	ms.ms_seed_mask_root_str = seed_mask_root_str;
	ms.ms_seed_mask_root_str_len = seed_mask_root_str ? strlen(seed_mask_root_str) + 1 : 0;
	ms.ms_u1_ids = u1_ids;
	ms.ms_u1_len = u1_len;
	ms.ms_u2_ids = u2_ids;
	ms.ms_u2_len = u2_len;
	ms.ms_my_client_id = my_client_id;
	ms.ms_threshold = threshold;
	ms.ms_output_vector = output_vector;
	ms.ms_out_max_len = out_max_len;
	status = sgx_ecall(eid, 3, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_generate_noise_from_seed(sgx_enclave_id_t eid, const char* seed_str, size_t len, long long* output)
{
	sgx_status_t status;
	ms_ecall_generate_noise_from_seed_t ms;
	ms.ms_seed_str = seed_str;
	ms.ms_seed_str_len = seed_str ? strlen(seed_str) + 1 : 0;
	ms.ms_len = len;
	ms.ms_output = output;
	status = sgx_ecall(eid, 4, &ocall_table_Enclave, &ms);
	return status;
}

