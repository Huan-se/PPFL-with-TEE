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
    long seed_mask_root, long seed_global_0, int client_id, 
    float k_weight, float n_ratio, 
    size_t model_len, int* ranges, size_t ranges_len, long long* output, size_t out_len
) {
    std::vector<float> grad;
    try {
        {
            std::lock_guard<std::mutex> lock(g_map_mutex);
            auto it = g_gradient_buffer.find(client_id);
            if (it == g_gradient_buffer.end()) {
                // Client data not found - fill zero and return
                for(size_t i=0; i<out_len; ++i) output[i] = 0;
                return;
            }
            grad = it->second; 
            g_gradient_buffer.erase(it); // Destroy state
        }
        
        if (grad.size() != model_len) {
             for(size_t i=0; i<out_len; ++i) output[i] = 0;
             return;
        }

        long seed_alpha = CryptoUtils::derive_seed(seed_mask_root, "alpha", 0);
        long seed_beta  = CryptoUtils::derive_seed(seed_mask_root, "beta", client_id);
        long seed_global_final = (seed_global_0 + seed_alpha) & 0x7FFFFFFF;

        DeterministicRandom rng_global(seed_global_final);
        DeterministicRandom rng_beta(seed_beta);

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
                    long long term_grad = (long long)(g_val * k_weight * SCALE);
                    long long vec_g = rng_global.next_mask_mod();
                    long long term_g = (long long)(vec_g * n_ratio);
                    long long vec_b = rng_beta.next_mask_mod();
                    unsigned __int128 sum = (unsigned __int128)term_grad + term_g + vec_b;
                    output[current_out_idx + i] = (long long)(sum % MOD);
                }
                current_out_idx += curr;
                offset += curr;
            }
        }
    } catch (...) {
        printf("[Enclave] Exception in generate_masked!\n");
        for(size_t i=0; i<out_len; ++i) output[i] = 0;
    }
}

void ecall_get_vector_shares_dynamic(
    long seed_sss, long seed_mask_root, 
    int target_client_id, int threshold, int total_clients, 
    struct SharePackage* output_shares
) {
    long secret_alpha = CryptoUtils::derive_seed(seed_mask_root, "alpha", 0);
    long secret_beta  = CryptoUtils::derive_seed(seed_mask_root, "beta", target_client_id);
    DeterministicRandom rng(seed_sss);
    struct TempCoeff { long long a; long long b; };
    std::vector<TempCoeff> poly(threshold);
    poly[0].a = secret_alpha; poly[0].b = secret_beta;
    for (int i = 1; i < threshold; ++i) {
        poly[i].a = rng.next_mask_mod(); poly[i].b = rng.next_mask_mod();
    }
    for (int x = 1; x <= total_clients; ++x) {
        unsigned __int128 res_a = poly[0].a; unsigned __int128 res_b = poly[0].b;
        unsigned __int128 x_pow = x;
        for (int i = 1; i < threshold; ++i) {
            res_a = (res_a + (unsigned __int128)poly[i].a * x_pow) % MOD;
            res_b = (res_b + (unsigned __int128)poly[i].b * x_pow) % MOD;
            x_pow = (x_pow * x) % MOD;
        }
        output_shares[x-1].share_alpha = (double)(uint64_t)res_a;
        output_shares[x-1].share_beta  = (double)(uint64_t)res_b;
    }
}