/* Enclave/Enclave.cpp */
#include "Enclave_t.h"
#include "sgx_trts.h"
#include "sgx_tcrypto.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

// ==========================================
// 1. 基础环境补丁 (Patch for std::rand & printf)
// ==========================================
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

// 代理 printf，通过 OCALL 输出到 App
int printf(const char *fmt, ...) {
    char buf[BUFSIZ] = { '\0' };
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, BUFSIZ, fmt, ap);
    va_end(ap);
    ocall_print_string(buf);
    return 0;
}

#include <string>
#include <vector>
#include <map>
#include <random>
#include <cmath>
#include <algorithm> 
#include <mutex>
#include <Eigen/Dense>

// ==========================================
// 2. 常量与全局变量
// ==========================================
#define CHUNK_SIZE 4096
const int64_t MOD = 9223372036854775783;
const double SCALE = 100000000.0; 
const uint64_t N_MASK = 0xFFFFFFFFFFFF; 

// 梯度缓存：持久化存储 Phase 2 计算的梯度，供 Phase 4 使用
static std::map<int, std::vector<float>> g_gradient_buffer;
static std::mutex g_map_mutex;

// ==========================================
// 3. 辅助工具类
// ==========================================
long parse_long(const char* str) {
    if (!str) return 0;
    char* end;
    return std::strtol(str, &end, 10);
}

float parse_float(const char* str) {
    if (!str) return 0.0f;
    char* end;
    return std::strtof(str, &end);
}

class MathUtils {
public:
    // 强制使用 __int128 防止模乘溢出
    static long long safe_mod_mul(long long a, long long b, long long m = MOD) {
        unsigned __int128 ua = (a >= 0) ? (unsigned __int128)a : (unsigned __int128)(a + m);
        unsigned __int128 ub = (b >= 0) ? (unsigned __int128)b : (unsigned __int128)(b + m);
        unsigned __int128 res = (ua * ub) % (unsigned __int128)m;
        return (long long)res;
    }

    // 扩展欧几里得算法求逆元
    static long long extended_gcd(long long a, long long b, long long &x, long long &y) {
        if (a == 0) { x = 0; y = 1; return b; }
        long long x1, y1;
        long long gcd = extended_gcd(b % a, a, x1, y1);
        x = y1 - (b / a) * x1;
        y = x1;
        return gcd;
    }

    static long long mod_inverse(long long a, long long m = MOD) {
        long long x, y;
        long long g = extended_gcd(a, m, x, y);
        if (g != 1) return 0; 
        return (x % m + m) % m;
    }
};

class CryptoUtils {
public:
    static long derive_seed(long root, const char* purpose, int id) {
        std::string s = std::to_string(root) + purpose + std::to_string(id);
        sgx_sha256_hash_t hash_output;
        sgx_sha256_msg((const uint8_t*)s.c_str(), (uint32_t)s.length(), &hash_output);
        uint32_t seed_val;
        memcpy(&seed_val, hash_output, sizeof(uint32_t));
        return (long)(seed_val & 0x7FFFFFFF);
    }
};

class DeterministicRandom {
private: std::mt19937 gen;
public:
    DeterministicRandom(long seed) : gen((unsigned int)seed) {}
    
    // 生成全范围随机数 (用于掩码)
    long long next_mask_mod() { 
        uint32_t low = gen(); uint32_t high = gen();
        uint64_t val = ((uint64_t)high << 32) | low;
        return (long long)(val % MOD); 
    }
    
    // 生成受限随机数 (用于 n_i)
    long long next_n_val() { 
        uint32_t low = gen(); uint32_t high = gen();
        uint64_t val = ((uint64_t)high << 32) | low;
        return (long long)(val & N_MASK); 
    }

    // 生成高斯分布 (用于投影)
    float next_normal() {
        float u1 = (gen() + 0.5f) / 4294967296.0f;
        float u2 = (gen() + 0.5f) / 4294967296.0f;
        return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.283185307f * u2);
    }
};

// ==========================================
// 4. ECALL 实现
// ==========================================

/* * Phase 2: 计算梯度并缓存 (Prepare Gradient)
 * 逻辑：Gradient = w_new - w_old
 */
void ecall_prepare_gradient(
    int client_id, const char* proj_seed_str,
    float* w_new, float* w_old, size_t model_len, 
    int* ranges, size_t ranges_len, float* output_proj, size_t out_len
) {
    long proj_seed = parse_long(proj_seed_str);
    try {
        std::vector<float> full_gradient;
        full_gradient.reserve(model_len);
        
        // [计算梯度]
        for(size_t i = 0; i < model_len; ++i) {
            full_gradient.push_back(w_new[i] - w_old[i]);
        }
        
        // [存入缓存]
        {
            std::lock_guard<std::mutex> lock(g_map_mutex);
            g_gradient_buffer[client_id] = full_gradient;
        }

        // [投影逻辑] (此处省略复杂投影，仅占位，重点在后续 SSS)
        DeterministicRandom rng(proj_seed);
        for(size_t i=0; i<out_len; ++i) output_proj[i] = rng.next_normal(); 

    } catch (...) {
        printf("[Enclave Error] OOM or Exception in prepare_gradient!\n");
    }
}

/* * Phase 4: 生成加掩码梯度 (Generate Masked Gradient)
 * 逻辑：C_i = G + c_i * M + B_i
 */
void ecall_generate_masked_gradient_dynamic(
    const char* seed_mask_root_str, const char* seed_global_0_str,
    int client_id, int* active_ids, size_t active_count, const char* k_weight_str,
    size_t model_len, int* ranges, size_t ranges_len, long long* output, size_t out_len
) {
    long seed_mask_root = parse_long(seed_mask_root_str);
    long seed_global_0 = parse_long(seed_global_0_str);
    float k_weight = parse_float(k_weight_str);

    std::vector<float> grad;
    try {
        std::lock_guard<std::mutex> lock(g_map_mutex);
        if (g_gradient_buffer.find(client_id) == g_gradient_buffer.end()) {
            printf("[Enclave ERROR] Gradient not found for client %d. Did Phase 2 run?\n", client_id);
            for(size_t i=0; i<out_len; ++i) output[i] = 0;
            return;
        }
        grad = g_gradient_buffer[client_id];
        // g_gradient_buffer.erase(client_id); // 可选：阅后即焚节省内存
    } catch(...) { return; }

    // 1. 计算系数 c_i = n_i / Sum(n_j)
    long long n_sum = 0;
    long long my_n_val = 0;
    
    // 遍历所有 Active 用户计算分母
    for (size_t k = 0; k < active_count; ++k) {
        int other_id = active_ids[k];
        long seed_n_other = CryptoUtils::derive_seed(seed_mask_root, "n_seq", other_id);
        DeterministicRandom rng_n(seed_n_other);
        long long n_val = rng_n.next_n_val();
        n_sum += n_val; 
        
        if (other_id == client_id) my_n_val = n_val;
    }
    n_sum %= MOD;
    long long inv_sum = MathUtils::mod_inverse(n_sum, MOD);
    long long c_i = MathUtils::safe_mod_mul(my_n_val, inv_sum, MOD);

    // 2. 准备掩码生成器
    long seed_alpha = CryptoUtils::derive_seed(seed_mask_root, "alpha", 0);
    long seed_M = (seed_global_0 + seed_alpha) & 0x7FFFFFFF;
    DeterministicRandom rng_M(seed_M); // 全局掩码

    long seed_beta = CryptoUtils::derive_seed(seed_mask_root, "beta", client_id);
    DeterministicRandom rng_B(seed_beta); // 私有掩码

    // 3. 加密循环
    size_t cur = 0;
    for (size_t r = 0; r < ranges_len; r += 2) {
        int start = ranges[r];
        int len = ranges[r+1];
        if (start < 0 || start + len > (int)model_len) continue;

        for(int i=0; i<len; ++i) {
            if(cur >= out_len) break;
            
            // 量化
            float g = grad[start+i];
            long long G = (long long)(g * k_weight * SCALE);
            G = (G % MOD + MOD) % MOD;
            
            // 掩码
            long long M = rng_M.next_mask_mod();
            long long B = rng_B.next_mask_mod();
            long long tM = MathUtils::safe_mod_mul(c_i, M, MOD);
            
            // 组合: C = G + c_i*M + B
            unsigned __int128 sum = (unsigned __int128)G + tM + B;
            output[cur++] = (long long)(sum % MOD);
        }
    }
}

/*
 * Phase 5: 向量化秘密共享 (Vector SSS)
 * 严格实现：构造 S = [Delta, Alpha, Beta...]，计算差值，生成 Shares
 */
void ecall_get_vector_shares_dynamic(
    const char* seed_sss_str, const char* seed_mask_root_str, 
    int* u1_ids, size_t u1_len, // Active Users (本轮应到)
    int* u2_ids, size_t u2_len, // Online Users (实际提交)
    int my_client_id, int threshold, 
    long long* output_vector, size_t out_max_len
) {
    long seed_sss = parse_long(seed_sss_str);
    long seed_mask_root = parse_long(seed_mask_root_str);

    try {
        // ---------------------------------------------------
        // 步骤 1: 计算掉线补偿值 (Delta)
        // Formula: Delta = Sum(n_drop) / Sum(n_all)
        // ---------------------------------------------------
        
        // A. 分母: Sum(n_all) based on U1
        long long n_sum_all = 0;
        for(size_t i=0; i<u1_len; ++i) {
            long s_n = CryptoUtils::derive_seed(seed_mask_root, "n_seq", u1_ids[i]);
            DeterministicRandom rng(s_n);
            n_sum_all += rng.next_n_val();
        }
        n_sum_all %= MOD;
        long long inv_sum_all = MathUtils::mod_inverse(n_sum_all, MOD);

        // B. 分子: Sum(n_drop) based on U1 \ U2 (Set Difference)
        long long n_sum_drop = 0;
        std::vector<int> u2_vec(u2_ids, u2_ids + u2_len);
        std::vector<int> dropped_ids; // 仅用于调试

        for (size_t i=0; i<u1_len; ++i) {
            int uid = u1_ids[i];
            
            // 检查 uid 是否在 u2_vec 中
            bool is_online = false;
            for (int alive : u2_vec) {
                if (uid == alive) {
                    is_online = true;
                    break;
                }
            }
            
            // 如果不在 U2 中，则是掉线用户，计入分子
            if (!is_online) {
                long s_n = CryptoUtils::derive_seed(seed_mask_root, "n_seq", uid);
                DeterministicRandom rng(s_n);
                n_sum_drop += rng.next_n_val();
                dropped_ids.push_back(uid);
            }
        }
        n_sum_drop %= MOD;

        // C. 计算 Delta
        long long delta = MathUtils::safe_mod_mul(n_sum_drop, inv_sum_all, MOD);

        // [DEBUG]
        if (my_client_id == 0) {
            printf("[Enclave SSS] Delta Calculation:\n");
            printf("  > U1 (Active): %lu, U2 (Online): %lu\n", u1_len, u2_len);
            printf("  > Dropped: %lu\n", dropped_ids.size());
            printf("  > n_sum_drop: %lld\n", n_sum_drop);
            printf("  > Calculated Delta: %lld\n", delta);
        }

        // ---------------------------------------------------
        // 步骤 2: 构造秘密向量 S
        // S = [Delta, Alpha, Beta_1, Beta_2, ..., Beta_N]
        // ---------------------------------------------------
        std::vector<long long> S;
        
        // S[0]: Delta
        S.push_back(delta);
        
        // S[1]: Alpha Seed (虽然大家都有，但通过 SSS 恢复可用于校验)
        long seed_alpha = CryptoUtils::derive_seed(seed_mask_root, "alpha", 0);
        S.push_back((long long)seed_alpha);
        
        // S[2...N]: 所有 U1 用户的 Beta Seeds
        // 必须包含 U1 中所有人的 Beta，因为 Server 恢复时需要知道掉线者的 Beta 吗？
        // 不，Server 需要消除的是 **Online (U2)** 用户的 B_i。
        // 但是为了通用性，向量中通常包含所有可能用户的 Beta，Server 恢复后按需取用。
        for (size_t i=0; i<u1_len; ++i) {
            int uid = u1_ids[i];
            long seed_beta = CryptoUtils::derive_seed(seed_mask_root, "beta", uid);
            S.push_back((long long)seed_beta);
        }

        if (out_max_len < S.size()) {
            printf("[Enclave Error] Output buffer too small! Need %lu, Got %lu\n", S.size(), out_max_len);
            return;
        }

        // ---------------------------------------------------
        // 步骤 3: 向量化生成 Shares
        // 对 S 中的每个元素 S[k]，生成多项式并求值
        // ---------------------------------------------------
        long long x_val = my_client_id + 1; // 当前 Client 的 x 坐标
        
        for (size_t k = 0; k < S.size(); ++k) {
            long long secret = S[k];
            
            // 派生该秘密对应的多项式系数种子
            // 种子 = Hash(SEED_sss || "poly" || k)
            long seed_poly = CryptoUtils::derive_seed(seed_sss, "poly_vec", (int)k);
            DeterministicRandom rng_poly(seed_poly);
            
            // 多项式求值: res = secret + a1*x + a2*x^2 ...
            long long res = secret;
            long long x_pow = x_val;
            
            for (int i = 1; i < threshold; ++i) {
                long long coeff = rng_poly.next_mask_mod();
                long long term = MathUtils::safe_mod_mul(coeff, x_pow, MOD);
                
                unsigned __int128 temp = (unsigned __int128)res + term;
                res = (long long)(temp % MOD);
                
                x_pow = MathUtils::safe_mod_mul(x_pow, x_val, MOD);
            }
            output_vector[k] = res;
        }
        
        // 填充剩余空间
        for (size_t k = S.size(); k < out_max_len; ++k) output_vector[k] = 0;

    } catch (...) {
        printf("[Enclave Error] Exception in get_vector_shares!\n");
    }
}

void ecall_generate_noise_from_seed(const char* seed_str, size_t len, long long* output) {
    long seed = parse_long(seed_str);
    try {
        DeterministicRandom rng(seed);
        for(size_t i=0; i<len; ++i) output[i] = rng.next_mask_mod();
    } catch (...) {}
}