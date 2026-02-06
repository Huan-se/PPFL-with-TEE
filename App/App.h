/* App/App.h */
#ifndef _APP_H_
#define _APP_H_

#if defined(__cplusplus)
extern "C" {
#endif

int tee_init(const char* enclave_filename);
void tee_destroy();

void tee_prepare_gradient(
    int client_id,
    long proj_seed, 
    float* w_new, 
    float* w_old, 
    int model_len, 
    int* ranges, 
    int ranges_len, 
    float* output_proj, 
    int out_len
);

void tee_generate_masked_gradient_dynamic(
    long seed_mask_root, 
    long seed_global_0, 
    int client_id, 
    int* active_ids, 
    int active_len, 
    float k_weight, 
    int model_len, 
    int* ranges, 
    int ranges_len, 
    long long* output, 
    int out_len
);

void tee_get_vector_shares_dynamic(
    long seed_sss, 
    long seed_mask_root, 
    int* u1_ids, 
    int u1_len, 
    int* u2_ids, 
    int u2_len, 
    int my_client_id, 
    int threshold, 
    long long* output_vector, 
    int out_max_len
);

void tee_generate_noise_from_seed(
    long seed, 
    int len, 
    long long* output
);

#if defined(__cplusplus)
}
#endif

#endif