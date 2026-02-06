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
const double SCALE = 1000000.0;
// 48位掩码，保证 n_sum 累加不溢出 int64
const uint64_t N_MASK = 0xFFFFFFFFFFFF; 

static std::map<int, std::vector<float>> g_gradient_buffer;
static std::mutex g_map_mutex;

class MathUtils {
public:
    static long long safe_mod_mul(long long a, long long b, long long m = MOD) {
        unsigned __int128 ua = (a >= 0) ? (unsigned __int128)a : (unsigned __int128)(a + m);
        unsigned __int128 ub = (b >= 0) ? (unsigned __int128)b : (unsigned __int128)(b + m);
        unsigned __int128 res = ua * ub;
        return (long long)(res % m);
    }

    static long long extended_gcd(long long a, long long b, long long &x, long long &y) {
        if (a == 0) {
            x = 0; y = 1;
            return b;
        }
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
    // 生成 [0, MOD) 的大随机数
    long long next_mask_mod() { return (long long)(next_u64() % MOD); }
    
    // [关键] 生成 48 位受限随机数，用于 n 值，防止累加溢出
    long long next_n_val() { 
        return (long long)(next_u64() & N_MASK); 
    }

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
        std::vector<float> full_gradient;
        full_gradient.reserve(model_len);
        for(size_t i = 0; i < model_len; ++i) {
            full_gradient.push_back(w_new[i] - w_old[i]);
        }
        
        {
            std::lock_guard<std::mutex> lock(g_map_mutex);
            g_gradient_buffer[client_id] = full_gradient;
        }

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
    int* active_ids,
    size_t active_count,
    float k_weight,
    size_t model_len, 
    int* ranges, 
    size_t ranges_len, 
    long long* output, 
    size_t out_len
) {
    std::vector<float> grad;
    try {
        {
            std::lock_guard<std::mutex> lock(g_map_mutex);
            auto it = g_gradient_buffer.find(client_id);
            if (it == g_gradient_buffer.end()) {
                for(size_t i=0; i<out_len; ++i) output[i] = 0;
                return;
            }
            grad = it->second; 
            g_gradient_buffer.erase(it);
        }
        
        if (grad.size() != model_len) {
             for(size_t i=0; i<out_len; ++i) output[i] = 0;
             return;
        }

        // --- 计算互掩码系数 ---
        long long n_sum = 0;
        long long my_n_val = 0;
        bool found_self = false;

        for (size_t k = 0; k < active_count; ++k) {
            int other_id = active_ids[k];
            long seed_n_other = CryptoUtils::derive_seed(seed_mask_root, "n_seq", other_id);
            DeterministicRandom rng_n(seed_n_other);
            
            // [V2 修正] 使用 next_n_val (48位)
            long long n_val = rng_n.next_n_val();
            n_sum += n_val; 
            
            if (other_id == client_id) {
                my_n_val = n_val;
                found_self = true;
            }
        }
        n_sum %= MOD;

        if (!found_self || n_sum == 0) {
            for(size_t i=0; i<out_len; ++i) output[i] = 0; 
            return;
        }

        long long inv_sum = MathUtils::mod_inverse(n_sum, MOD);
        if (inv_sum == 0) return;

        long long c_i = MathUtils::safe_mod_mul(my_n_val, inv_sum, MOD);

        // --- 准备 PRG ---
        long seed_alpha = CryptoUtils::derive_seed(seed_mask_root, "alpha", 0);
        long seed_M = (seed_global_0 + seed_alpha) & 0x7FFFFFFF;
        DeterministicRandom rng_M(seed_M);

        long seed_beta = CryptoUtils::derive_seed(seed_mask_root, "beta", client_id);
        DeterministicRandom rng_B(seed_beta);

        // --- 加密循环 ---
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
                    float g_val = grad[idx+i];
                    long long G_val = (long long)(g_val * k_weight * SCALE);
                    // 保证正数
                    G_val = (G_val % MOD + MOD) % MOD;

                    long long M_val = rng_M.next_mask_mod();
                    long long B_val = rng_B.next_mask_mod();

                    long long term_M = MathUtils::safe_mod_mul(c_i, M_val, MOD);

                    unsigned __int128 temp_sum = (unsigned __int128)G_val + term_M + B_val;
                    output[current_out_idx + i] = (long long)(temp_sum % MOD);
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
    int* u1_ids, size_t u1_len,
    int* u2_ids, size_t u2_len,
    int my_client_id,
    int threshold,
    long long* output_vector,
    size_t out_max_len
) {
    try {
        // 1. 重算 n_sum
        long long n_sum = 0;
        std::vector<int> u1_vec;
        for(size_t i=0; i<u1_len; ++i) {
            int uid = u1_ids[i];
            u1_vec.push_back(uid);
            
            long s_n = CryptoUtils::derive_seed(seed_mask_root, "n_seq", uid);
            DeterministicRandom rng(s_n);
            
            // [V2 修正] 必须与 generate_masked 一致，使用 next_n_val
            long long n_val = rng.next_n_val();
            n_sum += n_val;
        }
        n_sum %= MOD;
        
        long long inv_sum = MathUtils::mod_inverse(n_sum, MOD);
        if (inv_sum == 0) {
             for(size_t i=0; i<out_max_len; ++i) output_vector[i] = 0;
             return;
        }

        // 2. 计算 Delta
        long long n_drop_sum = 0;
        std::vector<int> u2_vec(u2_ids, u2_ids + u2_len);
        
        for (int uid : u1_vec) {
            bool is_active = false;
            for (int alive : u2_vec) {
                if (uid == alive) { is_active = true; break; }
            }
            if (!is_active) {
                long s_n = CryptoUtils::derive_seed(seed_mask_root, "n_seq", uid);
                DeterministicRandom rng(s_n);
                
                // [CRITICAL FIX] 之前错误使用了 next_mask_mod()，导致 n 值不匹配！
                // 必须修正为 next_n_val() 才能复现出正确的掉线用户 n 值
                long long n_val = rng.next_n_val();
                n_drop_sum += n_val;
            }
        }
        n_drop_sum %= MOD;
        
        long long delta = MathUtils::safe_mod_mul(n_drop_sum, inv_sum, MOD);

        // 3. 构建 Secret
        std::vector<long long> secrets;
        secrets.push_back(delta);
        
        long seed_alpha = CryptoUtils::derive_seed(seed_mask_root, "alpha", 0);
        secrets.push_back((long long)seed_alpha);

        std::sort(u2_vec.begin(), u2_vec.end()); 
        for (int uid : u2_vec) {
            long seed_beta = CryptoUtils::derive_seed(seed_mask_root, "beta", uid);
            secrets.push_back((long long)seed_beta);
        }
        
        if (out_max_len < secrets.size()) return;

        // 4. SSS 分片
        long long x_eval = my_client_id + 1;

        for (size_t k = 0; k < secrets.size(); ++k) {
            long long s_val = secrets[k];
            long seed_poly = CryptoUtils::derive_seed(seed_sss, "poly", (int)k);
            DeterministicRandom rng_poly(seed_poly);
            
            long long res = s_val;
            long long x_pow = x_eval; 
            
            for (int i = 1; i < threshold; ++i) {
                long long coeff = rng_poly.next_mask_mod();
                long long term = MathUtils::safe_mod_mul(coeff, x_pow, MOD);
                
                unsigned __int128 temp = (unsigned __int128)res + term;
                res = (long long)(temp % MOD);
                
                x_pow = MathUtils::safe_mod_mul(x_pow, x_eval, MOD);
            }
            output_vector[k] = res;
        }

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