/* App/App.h */
#ifndef _APP_H_
#define _APP_H_

#include <stddef.h> /* for size_t */

#if defined(__cplusplus)
extern "C" {
#endif

int tee_init(const char* enclave_path);
void tee_destroy();

// Phase 2
void tee_prepare_gradient(
    int client_id, 
    const char* proj_seed_str,  // [修改] 接收字符串
    float* w_new, float* w_old, size_t model_len, 
    int* ranges, size_t ranges_len, float* output_proj, size_t out_len
);

// Phase 4
void tee_generate_masked_gradient_dynamic(
    const char* seed_mask_root_str, // [修改] 接收字符串
    const char* seed_global_0_str,  // [修改]
    int client_id, 
    int* active_ids, size_t active_count,
    const char* k_weight_str,       // [修改]
    size_t model_len, 
    int* ranges, size_t ranges_len, 
    long long* output, size_t out_len
);

// Phase 5
void tee_get_vector_shares_dynamic(
    const char* seed_sss_str,       // [修改]
    const char* seed_mask_root_str, // [修改]
    int* u1_ids, size_t u1_len, 
    int* u2_ids, size_t u2_len, 
    int my_client_id, 
    int threshold, 
    long long* output_vector, 
    size_t out_max_len
);

// Noise
void tee_generate_noise_from_seed(
    const char* seed_str,           // [修改]
    size_t len, 
    long long* output
);

#if defined(__cplusplus)
}
#endif

#endif /* !_APP_H_ */