/* App/App.h */
#ifndef _APP_H_
#define _APP_H_

#if defined(__cplusplus)
extern "C" {
#endif

int tee_init(const char* enclave_filename);
void tee_destroy();

// 1. 投影接口
int tee_secure_aggregation(
    long seed, 
    float* w_new, 
    float* w_old, 
    int model_len, 
    int* ranges, 
    int ranges_len, 
    float* output, 
    int out_len
);

// 2. 双掩码梯度接口
int tee_generate_masked_gradient(
    long seed_r,
    long seed_b,
    float weight,
    float* w_new,
    float* w_old,
    int model_len,
    int* ranges,
    int ranges_len,
    float* output,
    int out_len
);

// 3. 恢复接口
int tee_get_recovery_share(
    long seed_sss,
    float secret_val,
    int threshold,
    int target_x,
    float* share_val
);

#if defined(__cplusplus)
}
#endif

#endif