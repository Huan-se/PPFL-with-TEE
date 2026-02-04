#include "Enclave_u.h"
#include <errno.h>

typedef struct ms_ecall_secure_aggregation_phase_t {
	long int ms_seed;
	float* ms_w_new;
	float* ms_w_old;
	size_t ms_model_len;
	int* ms_ranges;
	size_t ms_ranges_len;
	float* ms_output;
	size_t ms_out_len;
} ms_ecall_secure_aggregation_phase_t;

typedef struct ms_ecall_generate_masked_gradient_t {
	long int ms_seed_r;
	long int ms_seed_b;
	float ms_weight;
	float* ms_w_new;
	float* ms_w_old;
	size_t ms_model_len;
	int* ms_ranges;
	size_t ms_ranges_len;
	float* ms_output;
	size_t ms_out_len;
} ms_ecall_generate_masked_gradient_t;

typedef struct ms_ecall_get_recovery_share_t {
	long int ms_seed_sss;
	float ms_secret_val;
	int ms_threshold;
	int ms_target_x;
	float* ms_share_val;
} ms_ecall_get_recovery_share_t;

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
sgx_status_t ecall_secure_aggregation_phase(sgx_enclave_id_t eid, long int seed, float* w_new, float* w_old, size_t model_len, int* ranges, size_t ranges_len, float* output, size_t out_len)
{
	sgx_status_t status;
	ms_ecall_secure_aggregation_phase_t ms;
	ms.ms_seed = seed;
	ms.ms_w_new = w_new;
	ms.ms_w_old = w_old;
	ms.ms_model_len = model_len;
	ms.ms_ranges = ranges;
	ms.ms_ranges_len = ranges_len;
	ms.ms_output = output;
	ms.ms_out_len = out_len;
	status = sgx_ecall(eid, 0, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_generate_masked_gradient(sgx_enclave_id_t eid, long int seed_r, long int seed_b, float weight, float* w_new, float* w_old, size_t model_len, int* ranges, size_t ranges_len, float* output, size_t out_len)
{
	sgx_status_t status;
	ms_ecall_generate_masked_gradient_t ms;
	ms.ms_seed_r = seed_r;
	ms.ms_seed_b = seed_b;
	ms.ms_weight = weight;
	ms.ms_w_new = w_new;
	ms.ms_w_old = w_old;
	ms.ms_model_len = model_len;
	ms.ms_ranges = ranges;
	ms.ms_ranges_len = ranges_len;
	ms.ms_output = output;
	ms.ms_out_len = out_len;
	status = sgx_ecall(eid, 1, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_get_recovery_share(sgx_enclave_id_t eid, long int seed_sss, float secret_val, int threshold, int target_x, float* share_val)
{
	sgx_status_t status;
	ms_ecall_get_recovery_share_t ms;
	ms.ms_seed_sss = seed_sss;
	ms.ms_secret_val = secret_val;
	ms.ms_threshold = threshold;
	ms.ms_target_x = target_x;
	ms.ms_share_val = share_val;
	status = sgx_ecall(eid, 2, &ocall_table_Enclave, &ms);
	return status;
}

