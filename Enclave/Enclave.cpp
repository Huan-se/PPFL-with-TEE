/* Enclave/Enclave.cpp */
#include "Enclave_t.h"
#include "sgx_trts.h"
#include "sgx_tcrypto.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

// 1. 基础环境补丁
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

// [关键修改] 使用 128 位整数
typedef __int128_t int128;

#define CHUNK_SIZE 4096
const long long MOD = 9223372036854775783;
const double SCALE = 100000000.0; 
const uint64_t N_MASK = 0xFFFFFFFFFFFF; 

static std::map<int, std::vector<float>> g_gradient_buffer;
static std::mutex g_map_mutex;

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

// [新版数学库]
class MathUtils {
public:
    static long long safe_mod_add(long long a, long long b) {
        int128 ua = (int128)a;
        int128 ub = (int128)b;
        if (ua < 0) ua += MOD;
        if (ub < 0) ub += MOD;
        return (long long)((ua + ub) % (int128)MOD);
    }

    static long long safe_mod_sub(long long a, long long b) {
        int128 ua = (int128)a;
        int128 ub = (int128)b;
        if (ua < 0) ua += MOD;
        if (ub < 0) ub += MOD;
        int128 res = (ua - ub) % (int128)MOD;
        if (res < 0) res += MOD;
        return (long long)res;
    }

    static long long safe_mod_mul(long long a, long long b) {
        int128 ua = (int128)a;
        int128 ub = (int128)b;
        if (ua < 0) ua += MOD;
        if (ub < 0) ub += MOD;
        return (long long)((ua * ub) % (int128)MOD);
    }
    
    // 兼容重载
    static long long safe_mod_mul(long long a, long long b, long long m) {
        return safe_mod_mul(a, b);
    }

    static long long mod_inverse(long long n) {
        if (n == 0) return 0;
        int128 base = (int128)n;
        if (base < 0) base += MOD;
        int128 exp = (int128)MOD - 2; 
        int128 res = 1;
        base %= MOD;
        while (exp > 0) {
            if (exp % 2 == 1) res = (res * base) % MOD;
            base = (base * base) % MOD;
            exp /= 2;
        }
        return (long long)res;
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
    
    long long next_mask_mod() { 
        uint64_t limit = UINT64_MAX - (UINT64_MAX % MOD);
        uint64_t val;
        do {
            val = ((uint64_t)gen() << 32) | gen();
        } while (val >= limit);
        return (long long)(val % MOD);
    }
    
    long long next_n_val() { 
        uint32_t low = gen(); uint32_t high = gen();
        uint64_t val = ((uint64_t)high << 32) | low;
        return (long long)(val & N_MASK); 
    }

    float next_normal() {
        float u1 = (gen() + 0.5f) / 4294967296.0f;
        float u2 = (gen() + 0.5f) / 4294967296.0f;
        return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.283185307f * u2);
    }
};

// ---------------------------------------------------------
// ECALL 实现
// ---------------------------------------------------------

void ecall_prepare_gradient(
    int client_id, const char* proj_seed_str,
    float* w_new, float* w_old, size_t model_len, 
    int* ranges, size_t ranges_len, float* output_proj, size_t out_len
) {
    long proj_seed = parse_long(proj_seed_str);
    try {
        // 1. 计算完整梯度
        std::vector<float> full_gradient;
        full_gradient.reserve(model_len);
        for(size_t i = 0; i < model_len; ++i) {
            full_gradient.push_back(w_new[i] - w_old[i]);
        }
        
        // 2. 缓存梯度
        {
            std::lock_guard<std::mutex> lock(g_map_mutex);
            g_gradient_buffer[client_id] = full_gradient;
        }

        // 3. 流式投影 (V = P * G) - 使用 Eigen 加速
        // 这里恢复了真正的流式投影逻辑：
        // P 是通过 proj_seed 生成的随机高斯矩阵，
        // 我们分块生成 P 的行，并与梯度做点积。
        
        DeterministicRandom rng(proj_seed);
        Eigen::VectorXf rng_chunk(CHUNK_SIZE);

        for (size_t k = 0; k < out_len; ++k) {
            float dot_product = 0.0f;
            
            for (size_t r = 0; r < ranges_len; r += 2) {
                int start_idx = ranges[r];
                int block_len = ranges[r+1];
                if (start_idx < 0 || start_idx + block_len > (int)model_len) continue;
                
                int offset = 0;
                while (offset < block_len) {
                    int curr_size = std::min((int)CHUNK_SIZE, block_len - offset);
                    
                    // 生成 P 矩阵当前行的这一小段随机数
                    for(int i=0; i<curr_size; ++i) {
                        rng_chunk[i] = rng.next_normal();
                    }
                    
                    // 计算点积
                    Eigen::Map<Eigen::VectorXf> grad_segment(
                        full_gradient.data() + start_idx + offset, 
                        curr_size
                    );
                    dot_product += rng_chunk.head(curr_size).dot(grad_segment);
                    
                    offset += curr_size;
                }
            }
            // 直接输出点积结果 (无二值化，保留原始投影)
            output_proj[k] = dot_product;
        }

    } catch (...) {
        printf("[Enclave Error] OOM or Exception in prepare_gradient!\n");
        // 异常时清零输出，避免脏数据
        for(size_t i=0; i<out_len; ++i) output_proj[i] = 0.0f;
    }
}

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
            for(size_t i=0; i<out_len; ++i) output[i] = 0;
            return;
        }
        grad = g_gradient_buffer[client_id];
    } catch(...) { return; }

    // 1. 计算系数 c_i
    long long n_sum = 0;
    long long my_n_val = 0;
    for (size_t k = 0; k < active_count; ++k) {
        int other_id = active_ids[k];
        long seed_n_other = CryptoUtils::derive_seed(seed_mask_root, "n_seq", other_id);
        DeterministicRandom rng_n(seed_n_other);
        long long n_val = rng_n.next_n_val();
        n_sum = MathUtils::safe_mod_add(n_sum, n_val); 
        if (other_id == client_id) my_n_val = n_val;
    }
    long long inv_sum = MathUtils::mod_inverse(n_sum);
    long long c_i = MathUtils::safe_mod_mul(my_n_val, inv_sum);

    // 2. 掩码生成
    long seed_alpha = CryptoUtils::derive_seed(seed_mask_root, "alpha", 0);
    long seed_M = (seed_global_0 + seed_alpha) & 0x7FFFFFFF;
    DeterministicRandom rng_M(seed_M); 
    long seed_beta = CryptoUtils::derive_seed(seed_mask_root, "beta", client_id);
    DeterministicRandom rng_B(seed_beta); 

    // 3. 加密循环
    size_t cur = 0;
    for (size_t r = 0; r < ranges_len; r += 2) {
        int start = ranges[r];
        int len = ranges[r+1];
        if (start < 0 || start + len > (int)model_len) continue;

        for(int i=0; i<len; ++i) {
            if(cur >= out_len) break;
            
            float g = grad[start+i];
            long long G = (long long)(g * k_weight * SCALE);
            G = (G % MOD + MOD) % MOD;
            
            long long M = rng_M.next_mask_mod();
            long long B = rng_B.next_mask_mod();
            long long tM = MathUtils::safe_mod_mul(c_i, M);
            
            long long C = MathUtils::safe_mod_add(G, tM);
            C = MathUtils::safe_mod_add(C, B);
            
            output[cur++] = C;
        }
    }
}

void ecall_get_vector_shares_dynamic(
    const char* seed_sss_str, const char* seed_mask_root_str, 
    int* u1_ids, size_t u1_len, 
    int* u2_ids, size_t u2_len, 
    int my_client_id, int threshold, 
    long long* output_vector, size_t out_max_len
) {
    long seed_sss = parse_long(seed_sss_str);
    long seed_mask_root = parse_long(seed_mask_root_str);

    try {
        // Step 1: 计算 Delta
        long long n_sum_all = 0;
        for(size_t i=0; i<u1_len; ++i) {
            long s_n = CryptoUtils::derive_seed(seed_mask_root, "n_seq", u1_ids[i]);
            DeterministicRandom rng(s_n);
            n_sum_all = MathUtils::safe_mod_add(n_sum_all, rng.next_n_val());
        }
        long long inv_sum_all = MathUtils::mod_inverse(n_sum_all);

        long long n_sum_drop = 0;
        std::vector<int> u2_vec(u2_ids, u2_ids + u2_len);
        
        for (size_t i=0; i<u1_len; ++i) {
            int uid = u1_ids[i];
            bool is_online = false;
            for (int alive : u2_vec) if (uid == alive) is_online = true;
            
            if (!is_online) {
                long s_n = CryptoUtils::derive_seed(seed_mask_root, "n_seq", uid);
                DeterministicRandom rng(s_n);
                n_sum_drop = MathUtils::safe_mod_add(n_sum_drop, rng.next_n_val());
            }
        }
        long long delta = MathUtils::safe_mod_mul(n_sum_drop, inv_sum_all);

        // Step 2: 构造秘密向量 S
        std::vector<long long> S;
        S.push_back(delta);
        long seed_alpha = CryptoUtils::derive_seed(seed_mask_root, "alpha", 0);
        S.push_back((long long)seed_alpha);
        for (size_t i=0; i<u1_len; ++i) {
            int uid = u1_ids[i];
            long seed_beta = CryptoUtils::derive_seed(seed_mask_root, "beta", uid);
            S.push_back((long long)seed_beta);
        }

        // Step 3: 生成 Shares
        long long x_val = my_client_id + 1;
        for (size_t k = 0; k < S.size(); ++k) {
            long long secret = S[k];
            long seed_poly = CryptoUtils::derive_seed(seed_sss, "poly_vec", (int)k);
            DeterministicRandom rng_poly(seed_poly);
            
            long long res = secret;
            long long x_pow = x_val;
            
            for (int i = 1; i < threshold; ++i) {
                long long coeff = rng_poly.next_mask_mod();
                long long term = MathUtils::safe_mod_mul(coeff, x_pow);
                res = MathUtils::safe_mod_add(res, term);
                x_pow = MathUtils::safe_mod_mul(x_pow, x_val);
            }
            output_vector[k] = res;
        }
        for (size_t k = S.size(); k < out_max_len; ++k) output_vector[k] = 0;

    } catch (...) {}
}

void ecall_generate_noise_from_seed(const char* seed_str, size_t len, long long* output) {
    long seed = parse_long(seed_str);
    try {
        DeterministicRandom rng(seed);
        for(size_t i=0; i<len; ++i) output[i] = rng.next_mask_mod();
    } catch (...) {}
}