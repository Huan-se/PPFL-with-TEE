#include "Enclave_u.h"
#include <errno.h>

typedef struct ms_ecall_prepare_gradient_t {
	int ms_client_id;
	long int ms_proj_seed;
	float* ms_w_new;
	float* ms_w_old;
	size_t ms_model_len;
	int* ms_ranges;
	size_t ms_ranges_len;
	float* ms_output_proj;
	size_t ms_out_len;
} ms_ecall_prepare_gradient_t;

typedef struct ms_ecall_generate_masked_gradient_dynamic_t {
	long int ms_seed_mask_root;
	long int ms_seed_global_0;
	int ms_client_id;
	float ms_k_weight;
	float ms_n_ratio;
	size_t ms_model_len;
	int* ms_ranges;
	size_t ms_ranges_len;
	long long* ms_output;
	size_t ms_out_len;
} ms_ecall_generate_masked_gradient_dynamic_t;

typedef struct ms_ecall_get_vector_shares_dynamic_t {
	long int ms_seed_sss;
	long int ms_seed_mask_root;
	int ms_target_client_id;
	int ms_threshold;
	int ms_total_clients;
	struct SharePackage* ms_output_shares;
} ms_ecall_get_vector_shares_dynamic_t;

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
sgx_status_t ecall_prepare_gradient(sgx_enclave_id_t eid, int client_id, long int proj_seed, float* w_new, float* w_old, size_t model_len, int* ranges, size_t ranges_len, float* output_proj, size_t out_len)
{
	sgx_status_t status;
	ms_ecall_prepare_gradient_t ms;
	ms.ms_client_id = client_id;
	ms.ms_proj_seed = proj_seed;
	ms.ms_w_new = w_new;
	ms.ms_w_old = w_old;
	ms.ms_model_len = model_len;
	ms.ms_ranges = ranges;
	ms.ms_ranges_len = ranges_len;
	ms.ms_output_proj = output_proj;
	ms.ms_out_len = out_len;
	status = sgx_ecall(eid, 0, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_generate_masked_gradient_dynamic(sgx_enclave_id_t eid, long int seed_mask_root, long int seed_global_0, int client_id, float k_weight, float n_ratio, size_t model_len, int* ranges, size_t ranges_len, long long* output, size_t out_len)
{
	sgx_status_t status;
	ms_ecall_generate_masked_gradient_dynamic_t ms;
	ms.ms_seed_mask_root = seed_mask_root;
	ms.ms_seed_global_0 = seed_global_0;
	ms.ms_client_id = client_id;
	ms.ms_k_weight = k_weight;
	ms.ms_n_ratio = n_ratio;
	ms.ms_model_len = model_len;
	ms.ms_ranges = ranges;
	ms.ms_ranges_len = ranges_len;
	ms.ms_output = output;
	ms.ms_out_len = out_len;
	status = sgx_ecall(eid, 1, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_get_vector_shares_dynamic(sgx_enclave_id_t eid, long int seed_sss, long int seed_mask_root, int target_client_id, int threshold, int total_clients, struct SharePackage* output_shares)
{
	sgx_status_t status;
	ms_ecall_get_vector_shares_dynamic_t ms;
	ms.ms_seed_sss = seed_sss;
	ms.ms_seed_mask_root = seed_mask_root;
	ms.ms_target_client_id = target_client_id;
	ms.ms_threshold = threshold;
	ms.ms_total_clients = total_clients;
	ms.ms_output_shares = output_shares;
	status = sgx_ecall(eid, 2, &ocall_table_Enclave, &ms);
	return status;
}

