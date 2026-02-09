/* App/ServerCore.cpp */
#include <vector>
#include <random>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <string>
#include <iostream>
#include <openssl/sha.h> 
#include <cstdio> 
#include <map>

// [关键修改] 使用 128 位整数防止溢出
typedef __int128_t int128;
const long long MOD = 9223372036854775783;
static int g_core_verbose = 0;
#define LOG_DEBUG(fmt, ...) \
    do { if (g_core_verbose) printf("[ServerCore DEBUG] " fmt, ##__VA_ARGS__); } while (0)
// ---------------------------------------------------------
// 基础工具函数
// ---------------------------------------------------------
// void server_core_set_verbose(int level) {
//     g_core_verbose = level;
//     // printf("[ServerCore] Verbose level set to: %d\n", level);
// }
long long parse_long(const char* str) {
    if (!str) return 0;
    try { return std::stoll(str); } catch (...) { return 0; }
}
class CryptoUtils {
public:
    static long derive_seed(long root, const char* purpose, int id) {
        std::string s = std::to_string(root) + purpose + std::to_string(id);
        unsigned char hash[SHA256_DIGEST_LENGTH];
        SHA256((const unsigned char*)s.c_str(), s.length(), hash);
        uint32_t seed_val;
        std::memcpy(&seed_val, hash, sizeof(uint32_t));
        return (long)(seed_val & 0x7FFFFFFF);
    }
};

class MathUtils {
public:
    // 安全加法: (a + b) % m
    static long long safe_mod_add(long long a, long long b) {
        int128 ua = (int128)a;
        int128 ub = (int128)b;
        if (ua < 0) ua += MOD;
        if (ub < 0) ub += MOD;
        return (long long)((ua + ub) % (int128)MOD);
    }

    // 安全减法: (a - b) % m
    static long long safe_mod_sub(long long a, long long b) {
        int128 ua = (int128)a;
        int128 ub = (int128)b;
        if (ua < 0) ua += MOD;
        if (ub < 0) ub += MOD;
        
        int128 res = (ua - ub) % (int128)MOD;
        if (res < 0) res += MOD;
        return (long long)res;
    }

    // 安全乘法: (a * b) % m
    static long long safe_mod_mul(long long a, long long b) {
        int128 ua = (int128)a;
        int128 ub = (int128)b;
        if (ua < 0) ua += MOD;
        if (ub < 0) ub += MOD;
        return (long long)((ua * ub) % (int128)MOD);
    }

    // 兼容旧接口的重载 (忽略 m 参数)
    static long long safe_mod_mul(long long a, long long b, long long m) {
        return safe_mod_mul(a, b);
    }

    // [核心修复] 使用费马小定理求逆元 (a^(MOD-2) % MOD)
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
};

// ---------------------------------------------------------
// 核心算法: 拉格朗日插值 (求截距 L(0))
// ---------------------------------------------------------
long long lagrange_interpolate_zero(const std::vector<int>& x_coords, const std::vector<long long>& y_coords) {
    size_t k = x_coords.size();
    long long secret = 0;
    
    for (size_t j = 0; j < k; ++j) {
        long long num = 1; 
        long long den = 1;
        long long xj = x_coords[j];
        
        for (size_t m = 0; m < k; ++m) {
            if (m == j) continue;
            long long xm = x_coords[m];
            
            // num *= (0 - xm)
            long long term_num = MathUtils::safe_mod_sub(0, xm);
            num = MathUtils::safe_mod_mul(num, term_num);
            
            // den *= (xj - xm)
            long long term_den = MathUtils::safe_mod_sub(xj, xm);
            den = MathUtils::safe_mod_mul(den, term_den);
        }
        
        long long den_inv = MathUtils::mod_inverse(den);
        long long term = MathUtils::safe_mod_mul(y_coords[j], num);
        term = MathUtils::safe_mod_mul(term, den_inv);
        
        secret = MathUtils::safe_mod_add(secret, term);
    }
    return secret;
}

extern "C" {

void server_core_set_verbose(int level) {
    g_core_verbose = level;
}

// ---------------------------------------------------------
// 接口: 聚合与消去 (Server Core Aggregate & Unmask)
// ---------------------------------------------------------
void server_core_aggregate_and_unmask(
    const char* seed_mask_root_str,
    const char* seed_global_0_str,
    int* u1_ids, int u1_len, 
    int* u2_ids, int u2_len, 
    long long* shares_flat, 
    int vector_len,          
    long long* ciphers_flat, 
    int data_len,            
    long long* output_result 
) {
    long long seed_global_0 = parse_long(seed_global_0_str);

    // Step 1: 恢复秘密向量 S
    std::vector<long long> reconstructed_secrets(vector_len);
    std::vector<int> x_coords;
    for(int i=0; i<u2_len; ++i) x_coords.push_back(u2_ids[i] + 1);

    LOG_DEBUG("[ServerCore] Recovering Secret Vector (Len=%d) from %d clients...\n", vector_len, u2_len);

    for (int k = 0; k < vector_len; ++k) {
        std::vector<long long> y_coords;
        for (int i = 0; i < u2_len; ++i) {
            long long share = shares_flat[i * (size_t)vector_len + k];
            y_coords.push_back(share);
        }
        reconstructed_secrets[k] = lagrange_interpolate_zero(x_coords, y_coords);
    }

    // Step 2: 解析秘密向量
    long long delta = reconstructed_secrets[0];
    long long alpha_seed_rec = reconstructed_secrets[1];
    LOG_DEBUG("[ServerCore] RECONSTRUCTED DELTA: %lld\n", delta);

    std::map<int, long long> beta_map;
    for (int i = 0; i < u1_len; ++i) {
        if (2 + i < vector_len) {
            beta_map[u1_ids[i]] = reconstructed_secrets[2 + i];
        }
    }

    // Step 3: 准备消除噪声
    long seed_M = (seed_global_0 + alpha_seed_rec) & 0x7FFFFFFF;
    DeterministicRandom rng_M(seed_M);

    std::vector<DeterministicRandom> rng_B_list;
    for (int i = 0; i < u2_len; ++i) {
        int online_uid = u2_ids[i];
        if (beta_map.find(online_uid) != beta_map.end()) {
            long long s_b_long = beta_map[online_uid];
            rng_B_list.emplace_back((long)(s_b_long & 0x7FFFFFFF));
        }
    }

    long long coeff_M = MathUtils::safe_mod_sub(1, delta);
    LOG_DEBUG("[ServerCore] Coeff_M (1-Delta): %lld\n", coeff_M);

    // Step 4: 流式聚合与消去
    for (int k = 0; k < data_len; ++k) {
        // A. 生成噪声 N_k
        long long val_M = rng_M.next_mask_mod();
        long long term_M = MathUtils::safe_mod_mul(coeff_M, val_M);

        long long sum_B = 0;
        for (auto& rng : rng_B_list) {
            sum_B = MathUtils::safe_mod_add(sum_B, rng.next_mask_mod());
        }
        
        long long noise_k = MathUtils::safe_mod_add(term_M, sum_B);

        // B. 累加密文 Sum(C_k)
        long long sum_cipher_k = 0;
        for(int i=0; i<u2_len; ++i) {
            long long val = ciphers_flat[i * (size_t)data_len + k];
            sum_cipher_k = MathUtils::safe_mod_add(sum_cipher_k, val);
        }

        // C. 消去: Result = Sum(C) - N
        output_result[k] = MathUtils::safe_mod_sub(sum_cipher_k, noise_k);
        if (k < 5) {
            LOG_DEBUG("[ServerCore DEBUG] Aggregated Idx %d: ResultInt=%lld\n", k, output_result[k]);
        }
    }
}

} // extern "C"