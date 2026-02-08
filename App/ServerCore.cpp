/* App/ServerCore.cpp */
#include <vector>
#include <random>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <string>
#include <iostream>
#include <openssl/sha.h> 

const int64_t MOD = 9223372036854775783;
const uint64_t N_MASK = 0xFFFFFFFFFFFF; 

long long parse_long(const char* str) {
    if (!str) return 0;
    try { return std::stoll(str); } catch (...) { return 0; }
}

void write_long_to_buffer(long long val, char* buffer, size_t max_len) {
    if (!buffer || max_len == 0) return;
    std::string s = std::to_string(val);
    strncpy(buffer, s.c_str(), max_len - 1);
    buffer[max_len - 1] = '\0';
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
    static long long safe_mod_mul(long long a, long long b, long long m = MOD) {
        unsigned __int128 ua = (a >= 0) ? (unsigned __int128)a : (unsigned __int128)(a + m);
        unsigned __int128 ub = (b >= 0) ? (unsigned __int128)b : (unsigned __int128)(b + m);
        unsigned __int128 res = ua * ub;
        return (long long)(res % m);
    }
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

class DeterministicRandom {
private: std::mt19937 gen;
public:
    DeterministicRandom(long seed) : gen((unsigned int)seed) {}
    long long next_mask_mod() {
        uint32_t low = gen();
        uint32_t high = gen();
        uint64_t val = ((uint64_t)high << 32) | low;
        return (long long)(val % MOD);
    }
    long long next_n_val() { 
        uint32_t low = gen();
        uint32_t high = gen();
        uint64_t val = ((uint64_t)high << 32) | low;
        return (long long)(val & N_MASK); 
    }
};

extern "C" {

// 接口 1: 计算 Delta 和 n_sum (String IO)
void server_core_calc_secrets(
    const char* seed_mask_root_str,
    int* u1_ids, int u1_len,
    int* u2_ids, int u2_len,
    char* out_delta_str,
    char* out_n_sum_str,
    size_t buffer_len
) {
    long long seed_mask_root = parse_long(seed_mask_root_str);
    long long n_sum = 0;
    std::vector<int> u1_vec;
    for(int i=0; i<u1_len; ++i) {
        int uid = u1_ids[i];
        u1_vec.push_back(uid);
        long s_n = CryptoUtils::derive_seed(seed_mask_root, "n_seq", uid);
        DeterministicRandom rng(s_n);
        n_sum += rng.next_n_val();
    }
    n_sum %= MOD;
    write_long_to_buffer(n_sum, out_n_sum_str, buffer_len);

    long long inv_sum = MathUtils::mod_inverse(n_sum, MOD);
    if (inv_sum == 0) { 
        write_long_to_buffer(0, out_delta_str, buffer_len);
        return; 
    }

    long long n_drop_sum = 0;
    std::vector<int> u2_vec(u2_ids, u2_ids + u2_len);
    for (int uid : u1_vec) {
        bool is_active = false;
        for (int alive : u2_vec) if (uid == alive) is_active = true;
        if (!is_active) {
            long s_n = CryptoUtils::derive_seed(seed_mask_root, "n_seq", uid);
            DeterministicRandom rng(s_n);
            n_drop_sum += rng.next_n_val();
        }
    }
    n_drop_sum %= MOD;
    long long delta = MathUtils::safe_mod_mul(n_drop_sum, inv_sum, MOD);
    write_long_to_buffer(delta, out_delta_str, buffer_len);
}

// 接口 2: 生成噪声向量 (String IO)
void server_core_gen_noise_vector(
    const char* seed_mask_root_str,
    const char* seed_global_0_str,
    const char* delta_str,
    int* u2_ids, int u2_len,
    long long* output_noise,
    int data_len
) {
    long long seed_mask_root = parse_long(seed_mask_root_str);
    long long seed_global_0 = parse_long(seed_global_0_str);
    long long delta = parse_long(delta_str);

    long seed_alpha = CryptoUtils::derive_seed(seed_mask_root, "alpha", 0);
    long seed_M = (seed_global_0 + seed_alpha) & 0x7FFFFFFF;
    DeterministicRandom rng_M(seed_M);

    std::vector<DeterministicRandom> rng_B_list;
    for(int i=0; i<u2_len; ++i) {
        long s_b = CryptoUtils::derive_seed(seed_mask_root, "beta", u2_ids[i]);
        rng_B_list.emplace_back(s_b);
    }

    long long coeff_M = (1 - delta) % MOD;
    coeff_M = (coeff_M + MOD) % MOD;

    for (int k = 0; k < data_len; ++k) {
        long long val_M = rng_M.next_mask_mod();
        long long term_M = MathUtils::safe_mod_mul(coeff_M, val_M, MOD);
        long long sum_B = 0;
        for (auto& rng : rng_B_list) sum_B += rng.next_mask_mod();
        sum_B %= MOD;
        output_noise[k] = (term_M + sum_B) % MOD;
    }
}

} // extern "C"