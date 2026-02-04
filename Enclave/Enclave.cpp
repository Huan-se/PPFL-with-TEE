/* Enclave/Enclave.cpp */
#include "Enclave_t.h"
#include <sgx_trts.h>
#include <stdio.h>
#include <stdlib.h>

// === 1. 兼容性补丁 (必须保留以支持 Eigen) ===
extern "C" {
    int rand(void);
    void srand(unsigned int seed);
}
#ifndef RAND_MAX
#define RAND_MAX 2147483647
#endif
namespace std {
    using ::rand;
    using ::srand;
}

#include <vector>
#include <random>
#include <cmath>
#include <algorithm> 
#include <Eigen/Dense>

#define CHUNK_SIZE 4096

// === 2. 有限域配置 (SSS) ===
const long long SSS_MOD = 2147483647; // Mersenne Prime 2^31 - 1
const float SSS_SCALE = 1000000.0f;   // 缩放因子

// 打印函数
int printf(const char *fmt, ...) {
    char buf[BUFSIZ] = { '\0' };
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, BUFSIZ, fmt, ap);
    va_end(ap);
    ocall_print_string(buf);
    return 0;
}

// === 3. 有限域算术工具类 ===
class FiniteField {
public:
    static long long add(long long a, long long b) {
        return (a + b) % SSS_MOD;
    }
    static long long mul(long long a, long long b) {
        unsigned __int128 res = (unsigned __int128)a * b;
        return (long long)(res % SSS_MOD);
    }
    static long long from_float(float val) {
        long long scaled = (long long)(val * SSS_SCALE);
        if (scaled < 0) scaled = (scaled % SSS_MOD) + SSS_MOD;
        return scaled % SSS_MOD;
    }
    static float to_float(long long val) {
        if (val > SSS_MOD / 2) val = val - SSS_MOD;
        return (float)val / SSS_SCALE;
    }
    
    // 多项式求值 P(target_x)
    static float eval_poly(long long seed_sss, float secret, int threshold, int x) {
        // 使用 seed_sss 生成系数 a_1 ... a_{t-1}
        std::mt19937 gen(static_cast<unsigned int>(seed_sss));
        std::uniform_int_distribution<long long> dist(0, SSS_MOD - 1);

        long long res = from_float(secret); // a_0
        long long x_pow = x; 
        
        for (int i = 1; i < threshold; ++i) {
            long long a_i = dist(gen);
            long long term = mul(a_i, x_pow);
            res = add(res, term);
            x_pow = mul(x_pow, x);
        }
        return to_float(res);
    }
};

// ==========================================================
// ECALL 1: 投影生成 (保留原有功能)
// ==========================================================
void ecall_secure_aggregation_phase(
    long int seed,
    float* w_new,
    float* w_old,
    size_t model_len,
    int* ranges,
    size_t ranges_len,
    float* output,
    size_t out_len
) {
    std::mt19937 gen(static_cast<unsigned int>(seed));
    std::normal_distribution<float> dist(0.0f, 1.0f);
    Eigen::VectorXf proj_chunk(CHUNK_SIZE);

    for (size_t k = 0; k < out_len; ++k) {
        float dot_product = 0.0f;
        
        for (size_t r = 0; r < ranges_len; r += 2) {
            int start_idx = ranges[r];
            int block_len = ranges[r+1];
            if (start_idx < 0 || start_idx + block_len > (int)model_len) continue;

            int offset = 0;
            while (offset < block_len) {
                int current_chunk_size = std::min((int)CHUNK_SIZE, block_len - offset);
                
                // 生成投影向量
                for (int i = 0; i < current_chunk_size; ++i) proj_chunk[i] = dist(gen);

                int real_base_idx = start_idx + offset;
                Eigen::Map<Eigen::VectorXf> vec_new(w_new + real_base_idx, current_chunk_size);
                Eigen::Map<Eigen::VectorXf> vec_old(w_old + real_base_idx, current_chunk_size);
                
                auto vec_proj = proj_chunk.head(current_chunk_size);
                dot_product += (vec_new - vec_old).dot(vec_proj);
                offset += current_chunk_size;
            }
        }
        output[k] = dot_product;
    }
}

// ==========================================================
// ECALL 2: 双掩码梯度生成 (新增功能)
// ==========================================================
void ecall_generate_masked_gradient(
    long seed_r,
    long seed_b,
    float weight,
    float* w_new,
    float* w_old,
    size_t model_len,
    int* ranges,
    size_t ranges_len,
    float* output,
    size_t out_len
) {
    std::mt19937 gen_r(static_cast<unsigned int>(seed_r));
    std::mt19937 gen_b(static_cast<unsigned int>(seed_b));
    std::normal_distribution<float> dist(0.0f, 1.0f);

    size_t current_out_idx = 0;

    for (size_t r = 0; r < ranges_len; r += 2) {
        int start_idx = ranges[r];
        int block_len = ranges[r+1];
        if (start_idx < 0 || start_idx + block_len > (int)model_len) continue;

        int offset = 0;
        while (offset < block_len) {
            int current_chunk_size = std::min((int)CHUNK_SIZE, block_len - offset);
            int real_base_idx = start_idx + offset;

            // Map Data
            Eigen::Map<Eigen::VectorXf> vec_new(w_new + real_base_idx, current_chunk_size);
            Eigen::Map<Eigen::VectorXf> vec_old(w_old + real_base_idx, current_chunk_size);

            // Generate Masks On-the-fly
            Eigen::VectorXf mask_r(current_chunk_size);
            Eigen::VectorXf mask_b(current_chunk_size);
            for(int i=0; i<current_chunk_size; ++i) {
                mask_r[i] = dist(gen_r);
                mask_b[i] = dist(gen_b);
            }

            // Calc: k * (W_new - W_old) + R + B
            // 注意：这里我们做的是加法，如果互掩码需要消除，外部传来的 Seed_B 
            // 应该使得不同客户端生成的 mask_b 能够满足消除逻辑 (如正负互补，或者全局Delta消除)
            Eigen::VectorXf res_chunk = weight * (vec_new - vec_old) + mask_r + mask_b;

            // Write Output
            if (current_out_idx + current_chunk_size <= out_len) {
                 Eigen::Map<Eigen::VectorXf>(output + current_out_idx, current_chunk_size) = res_chunk;
                 current_out_idx += current_chunk_size;
            }
            offset += current_chunk_size;
        }
    }
}

// ==========================================================
// ECALL 3: 掉线恢复 SSS (新增功能)
// ==========================================================
void ecall_get_recovery_share(
    long seed_sss,
    float secret_val,
    int threshold,
    int target_x,
    float* share_val
) {
    *share_val = FiniteField::eval_poly(seed_sss, secret_val, threshold, target_x);
}