/* Enclave/Enclave.cpp */
#include "Enclave_t.h"
#include "sgx_trts.h"
#include "sgx_tcrypto.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {
    int rand(void);
    void srand(unsigned int seed);
}
#ifndef RAND_MAX
#define RAND_MAX 2147483647
#endif
namespace std { using ::rand; using ::srand; }

#include <string>
#include <vector>
#include <map>
#include <random>
#include <cmath>
#include <algorithm> 
#include <mutex>
#include <Eigen/Dense>

#define CHUNK_SIZE 4096
const int64_t MOD = 9223372036854775783;
const double SCALE = 10000.0;

// Thread-safe Global Buffer
static std::map<int, std::vector<float>> g_gradient_buffer;
static std::mutex g_map_mutex;

class MathUtils {
public:
    /**
     * 安全模乘: (a * b) % m
     * 使用 unsigned __int128 处理中间结果，防止 64 位整数溢出
     */
    static long long safe_mod_mul(long long a, long long b, long long m = MOD) {
        // 将输入转为正数处理（防止负数取模差异）
        unsigned __int128 ua = (a >= 0) ? (unsigned __int128)a : (unsigned __int128)(a + m);
        unsigned __int128 ub = (b >= 0) ? (unsigned __int128)b : (unsigned __int128)(b + m);
        
        unsigned __int128 res = ua * ub;
        return (long long)(res % m);
    }

    /**
     * 扩展欧几里得算法 (Extended GCD)
     * 求解: ax + by = gcd(a, b)
     * 返回值: gcd(a, b)
     * 引用参数 x, y 会被更新为解
     */
    static long long extended_gcd(long long a, long long b, long long &x, long long &y) {
        if (a == 0) {
            x = 0; 
            y = 1;
            return b;
        }
        long long x1, y1;
        long long gcd = extended_gcd(b % a, a, x1, y1);
        x = y1 - (b / a) * x1;
        y = x1;
        return gcd;
    }

    /**
     * 模逆运算 (Modular Inverse)
     * 计算 a^-1 mod m
     * 即求解 x 使得 (a * x) % m == 1
     */
    static long long mod_inverse(long long a, long long m = MOD) {
        long long x, y;
        long long g = extended_gcd(a, m, x, y);
        
        if (g != 1) {
            // 如果 gcd != 1，说明逆元不存在 (在 P 为大素数时，除非 a 是 P 的倍数，否则极少发生)
            // 返回 0 表示错误或无解
            return 0; 
        } else {
            // 确保结果调整为正数范围 [0, m-1]
            long long res = (x % m + m) % m;
            return res;
        }
    }
};

int printf(const char *fmt, ...) {
    char buf[BUFSIZ] = { '\0' };
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, BUFSIZ, fmt, ap);
    va_end(ap);
    ocall_print_string(buf);
    return 0;
}

class CryptoUtils {
public:
    static long derive_seed(long root, const char* purpose, int id) {
        std::string s = std::to_string(root) + purpose + std::to_string(id);
        sgx_sha256_hash_t hash_output;
        if (sgx_sha256_msg((const uint8_t*)s.c_str(), (uint32_t)s.length(), &hash_output) != SGX_SUCCESS) return 0;
        uint32_t seed_val;
        memcpy(&seed_val, hash_output, sizeof(uint32_t));
        return (long)(seed_val & 0x7FFFFFFF);
    }
};

class DeterministicRandom {
private: std::mt19937 gen;
public:
    DeterministicRandom(long seed) : gen((unsigned int)seed) {}
    uint32_t next_u32() { return gen(); }
    uint64_t next_u64() {
        uint32_t low = gen(); uint32_t high = gen();
        return ((uint64_t)high << 32) | low;
    }
    long long next_mask_mod() { return (long long)(next_u64() % MOD); }
    float next_uniform() { return (gen() + 0.5f) / 4294967296.0f; }
    float next_normal() {
        float u1 = next_uniform(); float u2 = next_uniform();
        float r = std::sqrt(-2.0f * std::log(u1)); float theta = 6.283185307f * u2;
        return r * std::cos(theta);
    }
};

void ecall_prepare_gradient(
    int client_id, long proj_seed, 
    float* w_new, float* w_old, size_t model_len, 
    int* ranges, size_t ranges_len, float* output_proj, size_t out_len
) {
    try {
        // 1. Calculate Gradient
        std::vector<float> full_gradient;
        full_gradient.reserve(model_len);
        for(size_t i = 0; i < model_len; ++i) {
            full_gradient.push_back(w_new[i] - w_old[i]);
        }
        
        // 2. Lock & Store
        {
            std::lock_guard<std::mutex> lock(g_map_mutex);
            g_gradient_buffer[client_id] = full_gradient;
        }

        // 3. Project
        DeterministicRandom rng(proj_seed);
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
                    // Safe Check
                    if (idx + curr > (int)full_gradient.size()) break;
                    
                    Eigen::Map<Eigen::VectorXf> grad_segment(full_gradient.data() + idx, curr);
                    dot_product += grad_segment.dot(proj_chunk.head(curr));
                    offset += curr;
                }
            }
            output_proj[k] = dot_product;
        }
    } catch (...) {
        printf("[Enclave] Exception in prepare_gradient!\n");
    }
}

void ecall_generate_masked_gradient_dynamic(
    long seed_mask_root, 
    long seed_global_0, 
    int client_id, 
    int* active_ids,      // [变更] 活跃客户端ID列表
    size_t active_count,  // [变更] 活跃客户端数量
    float k_weight,       // 服务端下发的聚合权重
    size_t model_len, 
    int* ranges, 
    size_t ranges_len, 
    long long* output, 
    size_t out_len
) {
    std::vector<float> grad;
    try {
        // 1. 获取并消耗梯度 (Stateful)
        {
            std::lock_guard<std::mutex> lock(g_map_mutex);
            auto it = g_gradient_buffer.find(client_id);
            if (it == g_gradient_buffer.end()) {
                for(size_t i=0; i<out_len; ++i) output[i] = 0;
                return;
            }
            grad = it->second; 
            g_gradient_buffer.erase(it); // 阅后即焚，防止重用
        }
        
        if (grad.size() != model_len) {
             for(size_t i=0; i<out_len; ++i) output[i] = 0;
             return;
        }

        // 2. [核心逻辑] 计算互掩码权重 c_i
        // 方案: c_i = n_i * (sum(n_j))^-1 mod P
        
        long long n_sum = 0;
        long long my_n_val = 0;
        bool found_self = false;

        // 遍历所有活跃客户端，恢复他们的 n_j 并求和
        for (size_t k = 0; k < active_count; ++k) {
            int other_id = active_ids[k];
            
            // 使用 seed_mask_root 和 client_id 派生种子
            long seed_n_other = CryptoUtils::derive_seed(seed_mask_root, "n_seq", other_id);
            DeterministicRandom rng_n(seed_n_other);
            
            // 生成 n_j (使用 next_mask_mod 确保在模 P 范围内)
            long long n_val = rng_n.next_mask_mod();
            
            n_sum = (n_sum + n_val) % MOD;
            
            if (other_id == client_id) {
                my_n_val = n_val;
                found_self = true;
            }
        }

        if (!found_self || n_sum == 0) {
            // 异常：自己不在活跃列表中，或 n_sum 为 0
            for(size_t i=0; i<out_len; ++i) output[i] = 0; 
            return;
        }

        // 计算总和的模逆: Inv = (N_sum)^-1
        long long inv_sum = MathUtils::mod_inverse(n_sum, MOD);
        if (inv_sum == 0) { /* 逆元不存在的极罕见情况 */ return; }

        // 计算当前客户端的互掩码系数: c_i = n_i * Inv
        long long c_i = MathUtils::safe_mod_mul(my_n_val, inv_sum, MOD);


        // 3. 准备掩码生成器
        
        // (A) 全局掩码生成器 PRG(M)
        // 种子 = seed_global_0 + alpha (alpha 派生自 mask_root)
        long seed_alpha = CryptoUtils::derive_seed(seed_mask_root, "alpha", 0);
        long seed_M = (seed_global_0 + seed_alpha) & 0x7FFFFFFF;
        DeterministicRandom rng_M(seed_M);

        // (B) 私有自掩码生成器 PRG(B)
        // 种子 = beta (派生自 mask_root + client_id)
        long seed_beta = CryptoUtils::derive_seed(seed_mask_root, "beta", client_id);
        DeterministicRandom rng_B(seed_beta);


        // 4. 加密循环: C_i = (k * g * S) + c_i * M + B
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
                    // (a) 定点化梯度 G = floor(grad * k * S)
                    float g_val = grad[idx+i];
                    long long G_val = (long long)(g_val * k_weight * SCALE);
                    // 处理负数：C++ % 操作符对负数保留符号，需转为正模数
                    // 简单做法：直接映射到 [0, P)
                    G_val = (G_val % MOD + MOD) % MOD;

                    // (b) 生成掩码值
                    long long M_val = rng_M.next_mask_mod(); // 全局掩码 M
                    long long B_val = rng_B.next_mask_mod(); // 自掩码 B

                    // (c) 计算互掩码项: Term_M = c_i * M
                    long long term_M = MathUtils::safe_mod_mul(c_i, M_val, MOD);

                    // (d) 最终求和: C = G + Term_M + B
                    long long res = (G_val + term_M) % MOD;
                    res = (res + B_val) % MOD;

                    output[current_out_idx + i] = res;
                }
                current_out_idx += curr;
                offset += curr;
            }
        }
    } catch (...) {
        printf("[Enclave] Exception in generate_masked_dynamic!\n");
        for(size_t i=0; i<out_len; ++i) output[i] = 0;
    }
}

void ecall_get_vector_shares_dynamic(
    long seed_sss,
    long seed_mask_root,
    int* u1_ids, size_t u1_len,          // [新增] Phase 4 的参与者 (用于重算 Inv)
    int* u2_ids, size_t u2_len,          // [新增] Phase 5 的存活者 (用于计算掉线和打包 Beta)
    int my_client_id,                    // 当前 TEE 的 ID (作为 SSS 的 x 坐标)
    int threshold,
    long long* output_vector,            // [变更] 输出的是 long long 类型的向量分片
    size_t out_max_len                   // 防止 buffer 越界
) {
    try {
        // 1. 重构 Phase 4 的上下文 (计算 Inv)
        // 必须与 Phase 4 的逻辑完全一致，以确保 Inv 相同
        long long n_sum = 0;
        std::vector<int> u1_vec;
        for(size_t i=0; i<u1_len; ++i) {
            int uid = u1_ids[i];
            u1_vec.push_back(uid);
            
            long s_n = CryptoUtils::derive_seed(seed_mask_root, "n_seq", uid);
            DeterministicRandom rng(s_n);
            n_sum = (n_sum + rng.next_mask_mod()) % MOD;
        }
        
        // 计算总和的模逆
        long long inv_sum = MathUtils::mod_inverse(n_sum, MOD);
        if (inv_sum == 0) {
             // 异常情况，填充 0 返回
             for(size_t i=0; i<out_max_len; ++i) output_vector[i] = 0;
             return;
        }

        // 2. 识别掉线用户并计算 Delta
        // Delta = (sum(n_drop) * Inv) % MOD
        long long n_drop_sum = 0;
        std::vector<int> u2_vec(u2_ids, u2_ids + u2_len);
        
        for (int uid : u1_vec) {
            // 检查 uid 是否在 u2 (存活列表) 中
            bool is_active = false;
            for (int alive : u2_vec) {
                if (uid == alive) { is_active = true; break; }
            }
            
            // 如果不在 U2 中，说明掉线了，将其 n 值加入 drop_sum
            if (!is_active) {
                long s_n = CryptoUtils::derive_seed(seed_mask_root, "n_seq", uid);
                DeterministicRandom rng(s_n);
                n_drop_sum = (n_drop_sum + rng.next_mask_mod()) % MOD;
            }
        }
        
        long long delta = MathUtils::safe_mod_mul(n_drop_sum, inv_sum, MOD);

        // 3. 构建秘密向量 S = [Delta, Seed_Alpha, Seed_Beta_u2_1, Seed_Beta_u2_2, ...]
        // 这个向量的顺序必须所有 TEE 保持一致
        std::vector<long long> secrets;
        
        // (1) Delta
        secrets.push_back(delta);
        
        // (2) Alpha Seed (注意类型转换)
        long seed_alpha = CryptoUtils::derive_seed(seed_mask_root, "alpha", 0);
        secrets.push_back((long long)seed_alpha);

        // (3) 所有在线用户的 Beta Seeds
        // 注意：u2_vec 必须在所有 TEE 均有序。Python 传进来时最好已排序，这里也可再排一次
        std::sort(u2_vec.begin(), u2_vec.end()); 
        for (int uid : u2_vec) {
            long seed_beta = CryptoUtils::derive_seed(seed_mask_root, "beta", uid);
            secrets.push_back((long long)seed_beta);
        }
        
        // 校验输出 buffer 大小
        if (out_max_len < secrets.size()) {
             // Buffer too small
             return; 
        }

        // 4. 执行 Shamir Secret Sharing (SSS)
        // 对 secrets 中的每一个元素 s_k，构造多项式 P_k(x) 并求值
        // x 坐标 = my_client_id + 1 (防止 ID=0 导致密钥泄露)
        long long x_eval = my_client_id + 1;

        for (size_t k = 0; k < secrets.size(); ++k) {
            long long s_val = secrets[k];
            
            // 使用 seed_sss 和 索引 k 派生该元素的系数生成器
            // 这样所有 TEE 对第 k 个秘密生成的随机系数是完全相同的
            long seed_poly = CryptoUtils::derive_seed(seed_sss, "poly", (int)k);
            DeterministicRandom rng_poly(seed_poly);
            
            // 多项式求值: y = s_val + a_1*x + a_2*x^2 ...
            long long res = s_val;
            long long x_pow = x_eval; 
            
            for (int i = 1; i < threshold; ++i) {
                long long coeff = rng_poly.next_mask_mod();
                long long term = MathUtils::safe_mod_mul(coeff, x_pow, MOD);
                res = (res + term) % MOD;
                
                // x^(i+1)
                x_pow = MathUtils::safe_mod_mul(x_pow, x_eval, MOD);
            }
            
            output_vector[k] = res;
        }

        // 填充剩余 buffer 为 0
        for (size_t k = secrets.size(); k < out_max_len; ++k) {
            output_vector[k] = 0;
        }

    } catch (...) {
        printf("[Enclave] Exception in get_vector_shares!\n");
    }
}

void ecall_generate_noise_from_seed(long seed, size_t len, long long* output) {
    try {
        DeterministicRandom rng(seed);
        for(size_t i=0; i<len; ++i) {
            output[i] = rng.next_mask_mod();
        }
    } catch (...) {
        for(size_t i=0; i<len; ++i) output[i] = 0;
    }
}