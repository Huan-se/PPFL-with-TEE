/* Enclave/Enclave.cpp */
#include "Enclave_t.h"
#include <sgx_trts.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm> 
#include <Eigen/Dense>

// --- 兼容性定义 ---
extern "C" {
    int rand(void);
    void srand(unsigned int seed);
}
#ifndef RAND_MAX
#define RAND_MAX 2147483647
#endif
namespace std { using ::rand; using ::srand; }

#define CHUNK_SIZE 4096

// --- 常量定义 ---
const int64_t MOD = 2147483647; // Mersenne Prime (2^31-1)
const double SCALE = 1000000.0; // 定点化缩放 1e6

// 打印辅助
int printf(const char *fmt, ...) {
    char buf[BUFSIZ] = { '\0' };
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, BUFSIZ, fmt, ap);
    va_end(ap);
    ocall_print_string(buf);
    return 0;
}

// --- 工具类: 确定性随机数与有限域 ---
class CryptoUtils {
public:
    // 简单的种子派生 (模拟 Hash)
    static long derive_seed(long root, const char* purpose, int id) {
        // 在真实 SGX 中应使用 SHA256
        // 这里演示用简单的混合哈希
        std::hash<std::string> hasher;
        std::string s = std::to_string(root) + purpose + std::to_string(id);
        return (long)hasher(s) & 0x7FFFFFFF; // 保持正数
    }

    // 有限域加法
    static long long add_mod(long long a, long long b) {
        return (a + b) % MOD;
    }
    // 有限域乘法
    static long long mul_mod(long long a, long long b) {
        return (a * b) % MOD;
    }
    // Float -> Int64 (Fixed Point)
    static long long to_fixed(float val) {
        return (long long)(val * SCALE);
    }
    // Int64 -> Float
    static float from_fixed(long long val) {
        return (float)val / SCALE;
    }
};

// --- 工具类: 随机数生成器 ---
class DeterministicRandom {
private:
    std::mt19937 gen;
public:
    DeterministicRandom(long seed) : gen((unsigned int)seed) {}
    
    uint32_t next_mask() { return gen(); }
    
    // 生成 Uniform [0, 1] 用于投影
    float next_uniform() { return (gen() + 0.5f) / 4294967296.0f; }
    
    // Box-Muller Normal
    float next_normal() {
        float u1 = next_uniform();
        float u2 = next_uniform();
        float r = std::sqrt(-2.0f * std::log(u1));
        float theta = 6.283185307f * u2;
        return r * std::cos(theta);
    }
};

// ==========================================================
// Phase 2: 投影 (LSH)
// ==========================================================
void ecall_secure_aggregation_phase(
    long seed, float* w_new, float* w_old, size_t model_len, 
    int* ranges, size_t ranges_len, float* output, size_t out_len
) {
    DeterministicRandom rng(seed);
    Eigen::VectorXf proj_chunk(CHUNK_SIZE);
    
    for (size_t k = 0; k < out_len; ++k) {
        float dot_product = 0.0f;
        for (size_t r = 0; r < ranges_len; r += 2) {
            int start_idx = ranges[r];
            int block_len = ranges[r+1];
            if (start_idx < 0 || start_idx + block_len > (int)model_len) continue;
            
            int offset = 0;
            while (offset < block_len) {
                int curr = std::min((int)CHUNK_SIZE, block_len - offset);
                for(int i=0; i<curr; ++i) proj_chunk[i] = rng.next_normal();
                
                int idx = start_idx + offset;
                Eigen::Map<Eigen::VectorXf> vn(w_new + idx, curr);
                Eigen::Map<Eigen::VectorXf> vo(w_old + idx, curr);
                dot_product += (vn - vo).dot(proj_chunk.head(curr));
                offset += curr;
            }
        }
        output[k] = dot_product;
    }
}

// ==========================================================
// Phase 4: 双掩码加噪 (Dynamic Derivation)
// ==========================================================
void ecall_generate_masked_gradient_dynamic(
    long seed_mask_root, long seed_global_0, int client_id, 
    float k_weight, float n_ratio, 
    float* w_new, float* w_old, size_t model_len, 
    int* ranges, size_t ranges_len, float* output, size_t out_len
) {
    // 1. 派生种子
    long seed_alpha = CryptoUtils::derive_seed(seed_mask_root, "alpha", 0);
    long seed_beta  = CryptoUtils::derive_seed(seed_mask_root, "beta", client_id);
    
    // 全局掩码种子 = Hash(w_0) + alpha
    // 简单模拟: XOR 或 加法
    long seed_global_final = (seed_global_0 + seed_alpha) & 0x7FFFFFFF;

    // 2. 初始化生成器
    DeterministicRandom rng_global(seed_global_final);
    DeterministicRandom rng_beta(seed_beta);

    // 3. 准备定点化参数
    long long k_fixed = CryptoUtils::to_fixed(k_weight);
    // 注意: n_ratio 是小数, 我们用 n_ratio * SCALE * Vector
    // 为了防止精度问题，这里我们将 n_ratio 视为 0~1 的 float 直接乘
    // 更好的方式是传入整数 n_i 和 sum_n，这里简化处理
    
    int32_t* output_ptr = reinterpret_cast<int32_t*>(output);
    size_t current_out_idx = 0;

    for (size_t r = 0; r < ranges_len; r += 2) {
        int start_idx = ranges[r];
        int block_len = ranges[r+1];
        if (start_idx < 0 || start_idx + block_len > (int)model_len) continue;

        int offset = 0;
        while (offset < block_len) {
            int curr = std::min((int)CHUNK_SIZE, block_len - offset);
            int idx = start_idx + offset;

            for (int i = 0; i < curr; ++i) {
                // A. 梯度 (定点化)
                float grad = w_new[idx+i] - w_old[idx+i];
                // C_i = k * w ...
                long long term_grad = (long long)(grad * k_weight * SCALE);
                
                // B. 全局掩码 (Vector)
                // Term = ratio * PRG(Global)
                long long vec_g = rng_global.next_mask() % MOD;
                long long term_g = (long long)(vec_g * n_ratio); // 简单的乘法
                
                // C. 自掩码 (Vector)
                long long vec_b = rng_beta.next_mask() % MOD;
                
                // D. 聚合
                long long sum = term_grad + term_g + vec_b;
                
                // 写入 (保持正数)
                output_ptr[current_out_idx + i] = (int32_t)((sum % MOD + MOD) % MOD);
            }
            current_out_idx += curr;
            offset += curr;
        }
    }
}

// ==========================================================
// Phase 5: 向量化秘密共享 (打包 Alpha 和 Beta)
// ==========================================================
void ecall_get_vector_shares_dynamic(
    long seed_sss, long seed_mask_root, 
    int target_client_id, int threshold, int total_clients, 
    SharePackage* output_shares
) {
    // 1. 实时派生出目标秘密
    // 这里的逻辑是：既然大家都有 SEED_mask，那么我(TEE)可以算出
    // 任何人的 beta。所以请求者可以说 "请帮我恢复 Client X 的 beta"
    
    long secret_alpha = CryptoUtils::derive_seed(seed_mask_root, "alpha", 0);
    long secret_beta  = CryptoUtils::derive_seed(seed_mask_root, "beta", target_client_id);
    
    // 2. 初始化 SSS 随机性
    DeterministicRandom rng(seed_sss);

    // 3. 准备系数向量
    std::vector<SecretPackage> coeffs(threshold);
    
    // a_0 = Secret
    coeffs[0].seed_alpha = secret_alpha % MOD;
    coeffs[0].seed_beta  = secret_beta  % MOD;

    // a_1 ... a_t = Random
    for (int i = 1; i < threshold; ++i) {
        coeffs[i].seed_alpha = rng.next_mask() % MOD;
        coeffs[i].seed_beta  = rng.next_mask() % MOD;
    }

    // 4. 计算分片
    for (int x = 1; x <= total_clients; ++x) {
        long long res_a = coeffs[0].seed_alpha;
        long long res_b = coeffs[0].seed_beta;
        long long x_pow = x;
        
        for (int i = 1; i < threshold; ++i) {
            res_a = (res_a + coeffs[i].seed_alpha * x_pow) % MOD;
            res_b = (res_b + coeffs[i].seed_beta  * x_pow) % MOD;
            x_pow = (x_pow * x) % MOD;
        }
        
        // 存为 float 传出 (实际是 int64)
        output_shares[x-1].share_alpha = (float)res_a;
        output_shares[x-1].share_beta  = (float)res_b;
    }
}