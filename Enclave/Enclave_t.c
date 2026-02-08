#include "Enclave_t.h"

#include "sgx_trts.h" /* for sgx_ocalloc, sgx_is_outside_enclave */
#include "sgx_lfence.h" /* for sgx_lfence */

#include <errno.h>
#include <mbusafecrt.h> /* for memcpy_s etc */
#include <stdlib.h> /* for malloc/free etc */

#define CHECK_REF_POINTER(ptr, siz) do {	\
	if (!(ptr) || ! sgx_is_outside_enclave((ptr), (siz)))	\
		return SGX_ERROR_INVALID_PARAMETER;\
} while (0)

#define CHECK_UNIQUE_POINTER(ptr, siz) do {	\
	if ((ptr) && ! sgx_is_outside_enclave((ptr), (siz)))	\
		return SGX_ERROR_INVALID_PARAMETER;\
} while (0)

#define CHECK_ENCLAVE_POINTER(ptr, siz) do {	\
	if ((ptr) && ! sgx_is_within_enclave((ptr), (siz)))	\
		return SGX_ERROR_INVALID_PARAMETER;\
} while (0)

#define ADD_ASSIGN_OVERFLOW(a, b) (	\
	((a) += (b)) < (b)	\
)


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

static sgx_status_t SGX_CDECL sgx_ecall_prepare_gradient(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_prepare_gradient_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_prepare_gradient_t* ms = SGX_CAST(ms_ecall_prepare_gradient_t*, pms);
	sgx_status_t status = SGX_SUCCESS;
	const char* _tmp_proj_seed_str = ms->ms_proj_seed_str;
	size_t _len_proj_seed_str = ms->ms_proj_seed_str_len ;
	char* _in_proj_seed_str = NULL;
	float* _tmp_w_new = ms->ms_w_new;
	size_t _tmp_model_len = ms->ms_model_len;
	size_t _len_w_new = _tmp_model_len * sizeof(float);
	float* _in_w_new = NULL;
	float* _tmp_w_old = ms->ms_w_old;
	size_t _len_w_old = _tmp_model_len * sizeof(float);
	float* _in_w_old = NULL;
	int* _tmp_ranges = ms->ms_ranges;
	size_t _tmp_ranges_len = ms->ms_ranges_len;
	size_t _len_ranges = _tmp_ranges_len * sizeof(int);
	int* _in_ranges = NULL;
	float* _tmp_output_proj = ms->ms_output_proj;
	size_t _tmp_out_len = ms->ms_out_len;
	size_t _len_output_proj = _tmp_out_len * sizeof(float);
	float* _in_output_proj = NULL;

	if (sizeof(*_tmp_w_new) != 0 &&
		(size_t)_tmp_model_len > (SIZE_MAX / sizeof(*_tmp_w_new))) {
		return SGX_ERROR_INVALID_PARAMETER;
	}

	if (sizeof(*_tmp_w_old) != 0 &&
		(size_t)_tmp_model_len > (SIZE_MAX / sizeof(*_tmp_w_old))) {
		return SGX_ERROR_INVALID_PARAMETER;
	}

	if (sizeof(*_tmp_ranges) != 0 &&
		(size_t)_tmp_ranges_len > (SIZE_MAX / sizeof(*_tmp_ranges))) {
		return SGX_ERROR_INVALID_PARAMETER;
	}

	if (sizeof(*_tmp_output_proj) != 0 &&
		(size_t)_tmp_out_len > (SIZE_MAX / sizeof(*_tmp_output_proj))) {
		return SGX_ERROR_INVALID_PARAMETER;
	}

	CHECK_UNIQUE_POINTER(_tmp_proj_seed_str, _len_proj_seed_str);
	CHECK_UNIQUE_POINTER(_tmp_w_new, _len_w_new);
	CHECK_UNIQUE_POINTER(_tmp_w_old, _len_w_old);
	CHECK_UNIQUE_POINTER(_tmp_ranges, _len_ranges);
	CHECK_UNIQUE_POINTER(_tmp_output_proj, _len_output_proj);

	//
	// fence after pointer checks
	//
	sgx_lfence();

	if (_tmp_proj_seed_str != NULL && _len_proj_seed_str != 0) {
		_in_proj_seed_str = (char*)malloc(_len_proj_seed_str);
		if (_in_proj_seed_str == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_proj_seed_str, _len_proj_seed_str, _tmp_proj_seed_str, _len_proj_seed_str)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

		_in_proj_seed_str[_len_proj_seed_str - 1] = '\0';
		if (_len_proj_seed_str != strlen(_in_proj_seed_str) + 1)
		{
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}
	if (_tmp_w_new != NULL && _len_w_new != 0) {
		if ( _len_w_new % sizeof(*_tmp_w_new) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		_in_w_new = (float*)malloc(_len_w_new);
		if (_in_w_new == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_w_new, _len_w_new, _tmp_w_new, _len_w_new)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}
	if (_tmp_w_old != NULL && _len_w_old != 0) {
		if ( _len_w_old % sizeof(*_tmp_w_old) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		_in_w_old = (float*)malloc(_len_w_old);
		if (_in_w_old == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_w_old, _len_w_old, _tmp_w_old, _len_w_old)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}
	if (_tmp_ranges != NULL && _len_ranges != 0) {
		if ( _len_ranges % sizeof(*_tmp_ranges) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		_in_ranges = (int*)malloc(_len_ranges);
		if (_in_ranges == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_ranges, _len_ranges, _tmp_ranges, _len_ranges)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}
	if (_tmp_output_proj != NULL && _len_output_proj != 0) {
		if ( _len_output_proj % sizeof(*_tmp_output_proj) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		if ((_in_output_proj = (float*)malloc(_len_output_proj)) == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_output_proj, 0, _len_output_proj);
	}

	ecall_prepare_gradient(ms->ms_client_id, (const char*)_in_proj_seed_str, _in_w_new, _in_w_old, _tmp_model_len, _in_ranges, _tmp_ranges_len, _in_output_proj, _tmp_out_len);
	if (_in_output_proj) {
		if (memcpy_s(_tmp_output_proj, _len_output_proj, _in_output_proj, _len_output_proj)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}

err:
	if (_in_proj_seed_str) free(_in_proj_seed_str);
	if (_in_w_new) free(_in_w_new);
	if (_in_w_old) free(_in_w_old);
	if (_in_ranges) free(_in_ranges);
	if (_in_output_proj) free(_in_output_proj);
	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_generate_masked_gradient_dynamic(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_generate_masked_gradient_dynamic_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_generate_masked_gradient_dynamic_t* ms = SGX_CAST(ms_ecall_generate_masked_gradient_dynamic_t*, pms);
	sgx_status_t status = SGX_SUCCESS;
	const char* _tmp_seed_mask_root_str = ms->ms_seed_mask_root_str;
	size_t _len_seed_mask_root_str = ms->ms_seed_mask_root_str_len ;
	char* _in_seed_mask_root_str = NULL;
	const char* _tmp_seed_global_0_str = ms->ms_seed_global_0_str;
	size_t _len_seed_global_0_str = ms->ms_seed_global_0_str_len ;
	char* _in_seed_global_0_str = NULL;
	int* _tmp_active_ids = ms->ms_active_ids;
	size_t _tmp_active_count = ms->ms_active_count;
	size_t _len_active_ids = _tmp_active_count * sizeof(int);
	int* _in_active_ids = NULL;
	const char* _tmp_k_weight_str = ms->ms_k_weight_str;
	size_t _len_k_weight_str = ms->ms_k_weight_str_len ;
	char* _in_k_weight_str = NULL;
	int* _tmp_ranges = ms->ms_ranges;
	size_t _tmp_ranges_len = ms->ms_ranges_len;
	size_t _len_ranges = _tmp_ranges_len * sizeof(int);
	int* _in_ranges = NULL;
	long long* _tmp_output = ms->ms_output;
	size_t _tmp_out_len = ms->ms_out_len;
	size_t _len_output = _tmp_out_len * sizeof(long long);
	long long* _in_output = NULL;

	if (sizeof(*_tmp_active_ids) != 0 &&
		(size_t)_tmp_active_count > (SIZE_MAX / sizeof(*_tmp_active_ids))) {
		return SGX_ERROR_INVALID_PARAMETER;
	}

	if (sizeof(*_tmp_ranges) != 0 &&
		(size_t)_tmp_ranges_len > (SIZE_MAX / sizeof(*_tmp_ranges))) {
		return SGX_ERROR_INVALID_PARAMETER;
	}

	if (sizeof(*_tmp_output) != 0 &&
		(size_t)_tmp_out_len > (SIZE_MAX / sizeof(*_tmp_output))) {
		return SGX_ERROR_INVALID_PARAMETER;
	}

	CHECK_UNIQUE_POINTER(_tmp_seed_mask_root_str, _len_seed_mask_root_str);
	CHECK_UNIQUE_POINTER(_tmp_seed_global_0_str, _len_seed_global_0_str);
	CHECK_UNIQUE_POINTER(_tmp_active_ids, _len_active_ids);
	CHECK_UNIQUE_POINTER(_tmp_k_weight_str, _len_k_weight_str);
	CHECK_UNIQUE_POINTER(_tmp_ranges, _len_ranges);
	CHECK_UNIQUE_POINTER(_tmp_output, _len_output);

	//
	// fence after pointer checks
	//
	sgx_lfence();

	if (_tmp_seed_mask_root_str != NULL && _len_seed_mask_root_str != 0) {
		_in_seed_mask_root_str = (char*)malloc(_len_seed_mask_root_str);
		if (_in_seed_mask_root_str == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_seed_mask_root_str, _len_seed_mask_root_str, _tmp_seed_mask_root_str, _len_seed_mask_root_str)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

		_in_seed_mask_root_str[_len_seed_mask_root_str - 1] = '\0';
		if (_len_seed_mask_root_str != strlen(_in_seed_mask_root_str) + 1)
		{
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}
	if (_tmp_seed_global_0_str != NULL && _len_seed_global_0_str != 0) {
		_in_seed_global_0_str = (char*)malloc(_len_seed_global_0_str);
		if (_in_seed_global_0_str == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_seed_global_0_str, _len_seed_global_0_str, _tmp_seed_global_0_str, _len_seed_global_0_str)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

		_in_seed_global_0_str[_len_seed_global_0_str - 1] = '\0';
		if (_len_seed_global_0_str != strlen(_in_seed_global_0_str) + 1)
		{
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}
	if (_tmp_active_ids != NULL && _len_active_ids != 0) {
		if ( _len_active_ids % sizeof(*_tmp_active_ids) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		_in_active_ids = (int*)malloc(_len_active_ids);
		if (_in_active_ids == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_active_ids, _len_active_ids, _tmp_active_ids, _len_active_ids)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}
	if (_tmp_k_weight_str != NULL && _len_k_weight_str != 0) {
		_in_k_weight_str = (char*)malloc(_len_k_weight_str);
		if (_in_k_weight_str == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_k_weight_str, _len_k_weight_str, _tmp_k_weight_str, _len_k_weight_str)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

		_in_k_weight_str[_len_k_weight_str - 1] = '\0';
		if (_len_k_weight_str != strlen(_in_k_weight_str) + 1)
		{
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}
	if (_tmp_ranges != NULL && _len_ranges != 0) {
		if ( _len_ranges % sizeof(*_tmp_ranges) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		_in_ranges = (int*)malloc(_len_ranges);
		if (_in_ranges == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_ranges, _len_ranges, _tmp_ranges, _len_ranges)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}
	if (_tmp_output != NULL && _len_output != 0) {
		if ( _len_output % sizeof(*_tmp_output) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		if ((_in_output = (long long*)malloc(_len_output)) == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_output, 0, _len_output);
	}

	ecall_generate_masked_gradient_dynamic((const char*)_in_seed_mask_root_str, (const char*)_in_seed_global_0_str, ms->ms_client_id, _in_active_ids, _tmp_active_count, (const char*)_in_k_weight_str, ms->ms_model_len, _in_ranges, _tmp_ranges_len, _in_output, _tmp_out_len);
	if (_in_output) {
		if (memcpy_s(_tmp_output, _len_output, _in_output, _len_output)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}

err:
	if (_in_seed_mask_root_str) free(_in_seed_mask_root_str);
	if (_in_seed_global_0_str) free(_in_seed_global_0_str);
	if (_in_active_ids) free(_in_active_ids);
	if (_in_k_weight_str) free(_in_k_weight_str);
	if (_in_ranges) free(_in_ranges);
	if (_in_output) free(_in_output);
	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_get_vector_shares_dynamic(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_get_vector_shares_dynamic_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_get_vector_shares_dynamic_t* ms = SGX_CAST(ms_ecall_get_vector_shares_dynamic_t*, pms);
	sgx_status_t status = SGX_SUCCESS;
	const char* _tmp_seed_sss_str = ms->ms_seed_sss_str;
	size_t _len_seed_sss_str = ms->ms_seed_sss_str_len ;
	char* _in_seed_sss_str = NULL;
	const char* _tmp_seed_mask_root_str = ms->ms_seed_mask_root_str;
	size_t _len_seed_mask_root_str = ms->ms_seed_mask_root_str_len ;
	char* _in_seed_mask_root_str = NULL;
	int* _tmp_u1_ids = ms->ms_u1_ids;
	size_t _tmp_u1_len = ms->ms_u1_len;
	size_t _len_u1_ids = _tmp_u1_len * sizeof(int);
	int* _in_u1_ids = NULL;
	int* _tmp_u2_ids = ms->ms_u2_ids;
	size_t _tmp_u2_len = ms->ms_u2_len;
	size_t _len_u2_ids = _tmp_u2_len * sizeof(int);
	int* _in_u2_ids = NULL;
	long long* _tmp_output_vector = ms->ms_output_vector;
	size_t _tmp_out_max_len = ms->ms_out_max_len;
	size_t _len_output_vector = _tmp_out_max_len * sizeof(long long);
	long long* _in_output_vector = NULL;

	if (sizeof(*_tmp_u1_ids) != 0 &&
		(size_t)_tmp_u1_len > (SIZE_MAX / sizeof(*_tmp_u1_ids))) {
		return SGX_ERROR_INVALID_PARAMETER;
	}

	if (sizeof(*_tmp_u2_ids) != 0 &&
		(size_t)_tmp_u2_len > (SIZE_MAX / sizeof(*_tmp_u2_ids))) {
		return SGX_ERROR_INVALID_PARAMETER;
	}

	if (sizeof(*_tmp_output_vector) != 0 &&
		(size_t)_tmp_out_max_len > (SIZE_MAX / sizeof(*_tmp_output_vector))) {
		return SGX_ERROR_INVALID_PARAMETER;
	}

	CHECK_UNIQUE_POINTER(_tmp_seed_sss_str, _len_seed_sss_str);
	CHECK_UNIQUE_POINTER(_tmp_seed_mask_root_str, _len_seed_mask_root_str);
	CHECK_UNIQUE_POINTER(_tmp_u1_ids, _len_u1_ids);
	CHECK_UNIQUE_POINTER(_tmp_u2_ids, _len_u2_ids);
	CHECK_UNIQUE_POINTER(_tmp_output_vector, _len_output_vector);

	//
	// fence after pointer checks
	//
	sgx_lfence();

	if (_tmp_seed_sss_str != NULL && _len_seed_sss_str != 0) {
		_in_seed_sss_str = (char*)malloc(_len_seed_sss_str);
		if (_in_seed_sss_str == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_seed_sss_str, _len_seed_sss_str, _tmp_seed_sss_str, _len_seed_sss_str)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

		_in_seed_sss_str[_len_seed_sss_str - 1] = '\0';
		if (_len_seed_sss_str != strlen(_in_seed_sss_str) + 1)
		{
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}
	if (_tmp_seed_mask_root_str != NULL && _len_seed_mask_root_str != 0) {
		_in_seed_mask_root_str = (char*)malloc(_len_seed_mask_root_str);
		if (_in_seed_mask_root_str == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_seed_mask_root_str, _len_seed_mask_root_str, _tmp_seed_mask_root_str, _len_seed_mask_root_str)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

		_in_seed_mask_root_str[_len_seed_mask_root_str - 1] = '\0';
		if (_len_seed_mask_root_str != strlen(_in_seed_mask_root_str) + 1)
		{
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}
	if (_tmp_u1_ids != NULL && _len_u1_ids != 0) {
		if ( _len_u1_ids % sizeof(*_tmp_u1_ids) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		_in_u1_ids = (int*)malloc(_len_u1_ids);
		if (_in_u1_ids == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_u1_ids, _len_u1_ids, _tmp_u1_ids, _len_u1_ids)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}
	if (_tmp_u2_ids != NULL && _len_u2_ids != 0) {
		if ( _len_u2_ids % sizeof(*_tmp_u2_ids) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		_in_u2_ids = (int*)malloc(_len_u2_ids);
		if (_in_u2_ids == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_u2_ids, _len_u2_ids, _tmp_u2_ids, _len_u2_ids)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}
	if (_tmp_output_vector != NULL && _len_output_vector != 0) {
		if ( _len_output_vector % sizeof(*_tmp_output_vector) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		if ((_in_output_vector = (long long*)malloc(_len_output_vector)) == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_output_vector, 0, _len_output_vector);
	}

	ecall_get_vector_shares_dynamic((const char*)_in_seed_sss_str, (const char*)_in_seed_mask_root_str, _in_u1_ids, _tmp_u1_len, _in_u2_ids, _tmp_u2_len, ms->ms_my_client_id, ms->ms_threshold, _in_output_vector, _tmp_out_max_len);
	if (_in_output_vector) {
		if (memcpy_s(_tmp_output_vector, _len_output_vector, _in_output_vector, _len_output_vector)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}

err:
	if (_in_seed_sss_str) free(_in_seed_sss_str);
	if (_in_seed_mask_root_str) free(_in_seed_mask_root_str);
	if (_in_u1_ids) free(_in_u1_ids);
	if (_in_u2_ids) free(_in_u2_ids);
	if (_in_output_vector) free(_in_output_vector);
	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_generate_noise_from_seed(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_generate_noise_from_seed_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_generate_noise_from_seed_t* ms = SGX_CAST(ms_ecall_generate_noise_from_seed_t*, pms);
	sgx_status_t status = SGX_SUCCESS;
	const char* _tmp_seed_str = ms->ms_seed_str;
	size_t _len_seed_str = ms->ms_seed_str_len ;
	char* _in_seed_str = NULL;
	long long* _tmp_output = ms->ms_output;
	size_t _tmp_len = ms->ms_len;
	size_t _len_output = _tmp_len * sizeof(long long);
	long long* _in_output = NULL;

	if (sizeof(*_tmp_output) != 0 &&
		(size_t)_tmp_len > (SIZE_MAX / sizeof(*_tmp_output))) {
		return SGX_ERROR_INVALID_PARAMETER;
	}

	CHECK_UNIQUE_POINTER(_tmp_seed_str, _len_seed_str);
	CHECK_UNIQUE_POINTER(_tmp_output, _len_output);

	//
	// fence after pointer checks
	//
	sgx_lfence();

	if (_tmp_seed_str != NULL && _len_seed_str != 0) {
		_in_seed_str = (char*)malloc(_len_seed_str);
		if (_in_seed_str == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_seed_str, _len_seed_str, _tmp_seed_str, _len_seed_str)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

		_in_seed_str[_len_seed_str - 1] = '\0';
		if (_len_seed_str != strlen(_in_seed_str) + 1)
		{
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}
	if (_tmp_output != NULL && _len_output != 0) {
		if ( _len_output % sizeof(*_tmp_output) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		if ((_in_output = (long long*)malloc(_len_output)) == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_output, 0, _len_output);
	}

	ecall_generate_noise_from_seed((const char*)_in_seed_str, _tmp_len, _in_output);
	if (_in_output) {
		if (memcpy_s(_tmp_output, _len_output, _in_output, _len_output)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}

err:
	if (_in_seed_str) free(_in_seed_str);
	if (_in_output) free(_in_output);
	return status;
}

SGX_EXTERNC const struct {
	size_t nr_ecall;
	struct {void* ecall_addr; uint8_t is_priv; uint8_t is_switchless;} ecall_table[4];
} g_ecall_table = {
	4,
	{
		{(void*)(uintptr_t)sgx_ecall_prepare_gradient, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_generate_masked_gradient_dynamic, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_get_vector_shares_dynamic, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_generate_noise_from_seed, 0, 0},
	}
};

SGX_EXTERNC const struct {
	size_t nr_ocall;
	uint8_t entry_table[6][4];
} g_dyn_entry_table = {
	6,
	{
		{0, 0, 0, 0, },
		{0, 0, 0, 0, },
		{0, 0, 0, 0, },
		{0, 0, 0, 0, },
		{0, 0, 0, 0, },
		{0, 0, 0, 0, },
	}
};


sgx_status_t SGX_CDECL ocall_print_string(const char* str)
{
	sgx_status_t status = SGX_SUCCESS;
	size_t _len_str = str ? strlen(str) + 1 : 0;

	ms_ocall_print_string_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_ocall_print_string_t);
	void *__tmp = NULL;


	CHECK_ENCLAVE_POINTER(str, _len_str);

	if (ADD_ASSIGN_OVERFLOW(ocalloc_size, (str != NULL) ? _len_str : 0))
		return SGX_ERROR_INVALID_PARAMETER;

	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_ocall_print_string_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_ocall_print_string_t));
	ocalloc_size -= sizeof(ms_ocall_print_string_t);

	if (str != NULL) {
		ms->ms_str = (const char*)__tmp;
		if (_len_str % sizeof(*str) != 0) {
			sgx_ocfree();
			return SGX_ERROR_INVALID_PARAMETER;
		}
		if (memcpy_s(__tmp, ocalloc_size, str, _len_str)) {
			sgx_ocfree();
			return SGX_ERROR_UNEXPECTED;
		}
		__tmp = (void *)((size_t)__tmp + _len_str);
		ocalloc_size -= _len_str;
	} else {
		ms->ms_str = NULL;
	}
	
	status = sgx_ocall(0, ms);

	if (status == SGX_SUCCESS) {
	}
	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL sgx_oc_cpuidex(int cpuinfo[4], int leaf, int subleaf)
{
	sgx_status_t status = SGX_SUCCESS;
	size_t _len_cpuinfo = 4 * sizeof(int);

	ms_sgx_oc_cpuidex_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_sgx_oc_cpuidex_t);
	void *__tmp = NULL;

	void *__tmp_cpuinfo = NULL;

	CHECK_ENCLAVE_POINTER(cpuinfo, _len_cpuinfo);

	if (ADD_ASSIGN_OVERFLOW(ocalloc_size, (cpuinfo != NULL) ? _len_cpuinfo : 0))
		return SGX_ERROR_INVALID_PARAMETER;

	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_sgx_oc_cpuidex_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_sgx_oc_cpuidex_t));
	ocalloc_size -= sizeof(ms_sgx_oc_cpuidex_t);

	if (cpuinfo != NULL) {
		ms->ms_cpuinfo = (int*)__tmp;
		__tmp_cpuinfo = __tmp;
		if (_len_cpuinfo % sizeof(*cpuinfo) != 0) {
			sgx_ocfree();
			return SGX_ERROR_INVALID_PARAMETER;
		}
		memset(__tmp_cpuinfo, 0, _len_cpuinfo);
		__tmp = (void *)((size_t)__tmp + _len_cpuinfo);
		ocalloc_size -= _len_cpuinfo;
	} else {
		ms->ms_cpuinfo = NULL;
	}
	
	ms->ms_leaf = leaf;
	ms->ms_subleaf = subleaf;
	status = sgx_ocall(1, ms);

	if (status == SGX_SUCCESS) {
		if (cpuinfo) {
			if (memcpy_s((void*)cpuinfo, _len_cpuinfo, __tmp_cpuinfo, _len_cpuinfo)) {
				sgx_ocfree();
				return SGX_ERROR_UNEXPECTED;
			}
		}
	}
	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL sgx_thread_wait_untrusted_event_ocall(int* retval, const void* self)
{
	sgx_status_t status = SGX_SUCCESS;

	ms_sgx_thread_wait_untrusted_event_ocall_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_sgx_thread_wait_untrusted_event_ocall_t);
	void *__tmp = NULL;


	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_sgx_thread_wait_untrusted_event_ocall_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_sgx_thread_wait_untrusted_event_ocall_t));
	ocalloc_size -= sizeof(ms_sgx_thread_wait_untrusted_event_ocall_t);

	ms->ms_self = self;
	status = sgx_ocall(2, ms);

	if (status == SGX_SUCCESS) {
		if (retval) *retval = ms->ms_retval;
	}
	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL sgx_thread_set_untrusted_event_ocall(int* retval, const void* waiter)
{
	sgx_status_t status = SGX_SUCCESS;

	ms_sgx_thread_set_untrusted_event_ocall_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_sgx_thread_set_untrusted_event_ocall_t);
	void *__tmp = NULL;


	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_sgx_thread_set_untrusted_event_ocall_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_sgx_thread_set_untrusted_event_ocall_t));
	ocalloc_size -= sizeof(ms_sgx_thread_set_untrusted_event_ocall_t);

	ms->ms_waiter = waiter;
	status = sgx_ocall(3, ms);

	if (status == SGX_SUCCESS) {
		if (retval) *retval = ms->ms_retval;
	}
	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL sgx_thread_setwait_untrusted_events_ocall(int* retval, const void* waiter, const void* self)
{
	sgx_status_t status = SGX_SUCCESS;

	ms_sgx_thread_setwait_untrusted_events_ocall_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_sgx_thread_setwait_untrusted_events_ocall_t);
	void *__tmp = NULL;


	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_sgx_thread_setwait_untrusted_events_ocall_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_sgx_thread_setwait_untrusted_events_ocall_t));
	ocalloc_size -= sizeof(ms_sgx_thread_setwait_untrusted_events_ocall_t);

	ms->ms_waiter = waiter;
	ms->ms_self = self;
	status = sgx_ocall(4, ms);

	if (status == SGX_SUCCESS) {
		if (retval) *retval = ms->ms_retval;
	}
	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL sgx_thread_set_multiple_untrusted_events_ocall(int* retval, const void** waiters, size_t total)
{
	sgx_status_t status = SGX_SUCCESS;
	size_t _len_waiters = total * sizeof(void*);

	ms_sgx_thread_set_multiple_untrusted_events_ocall_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_sgx_thread_set_multiple_untrusted_events_ocall_t);
	void *__tmp = NULL;


	CHECK_ENCLAVE_POINTER(waiters, _len_waiters);

	if (ADD_ASSIGN_OVERFLOW(ocalloc_size, (waiters != NULL) ? _len_waiters : 0))
		return SGX_ERROR_INVALID_PARAMETER;

	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_sgx_thread_set_multiple_untrusted_events_ocall_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_sgx_thread_set_multiple_untrusted_events_ocall_t));
	ocalloc_size -= sizeof(ms_sgx_thread_set_multiple_untrusted_events_ocall_t);

	if (waiters != NULL) {
		ms->ms_waiters = (const void**)__tmp;
		if (_len_waiters % sizeof(*waiters) != 0) {
			sgx_ocfree();
			return SGX_ERROR_INVALID_PARAMETER;
		}
		if (memcpy_s(__tmp, ocalloc_size, waiters, _len_waiters)) {
			sgx_ocfree();
			return SGX_ERROR_UNEXPECTED;
		}
		__tmp = (void *)((size_t)__tmp + _len_waiters);
		ocalloc_size -= _len_waiters;
	} else {
		ms->ms_waiters = NULL;
	}
	
	ms->ms_total = total;
	status = sgx_ocall(5, ms);

	if (status == SGX_SUCCESS) {
		if (retval) *retval = ms->ms_retval;
	}
	sgx_ocfree();
	return status;
}

