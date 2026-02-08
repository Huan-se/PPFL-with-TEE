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

// 使用大素数域
const int64_t MOD = 9223372036854775783;

// ---------------------------------------------------------
// 基础工具函数
// ---------------------------------------------------------

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
    // 强制使用 __int128 防止模乘溢出
    static long long safe_mod_mul(long long a, long long b, long long m = MOD) {
        unsigned __int128 ua = (a >= 0) ? (unsigned __int128)a : (unsigned __int128)(a + m);
        unsigned __int128 ub = (b >= 0) ? (unsigned __int128)b : (unsigned __int128)(b + m);
        unsigned __int128 res = (ua * ub) % (unsigned __int128)m;
        return (long long)res;
    }
    
    // 扩展欧几里得求逆
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
        uint32_t low = gen(); uint32_t high = gen();
        uint64_t val = ((uint64_t)high << 32) | low;
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
            
            // num *= (0 - xm) => -xm
            long long neg_xm = (MOD - xm) % MOD; 
            num = MathUtils::safe_mod_mul(num, neg_xm, MOD);
            
            // den *= (xj - xm)
            long long diff = (xj - xm) % MOD;
            if (diff < 0) diff += MOD;
            den = MathUtils::safe_mod_mul(den, diff, MOD);
        }
        
        long long den_inv = MathUtils::mod_inverse(den, MOD);
        long long term = MathUtils::safe_mod_mul(y_coords[j], num, MOD); // y_j * num
        term = MathUtils::safe_mod_mul(term, den_inv, MOD);             // term / den
        
        unsigned __int128 temp_sum = (unsigned __int128)secret + term;
        secret = (long long)(temp_sum % MOD);
    }
    return secret;
}

extern "C" {

// ---------------------------------------------------------
// 接口: 聚合与消去 (Server Core Aggregate & Unmask)
// ---------------------------------------------------------
void server_core_aggregate_and_unmask(
    const char* seed_mask_root_str,
    const char* seed_global_0_str,
    int* u1_ids, int u1_len, // 所有本轮参与者 (用于 Beta 映射)
    int* u2_ids, int u2_len, // 实际在线用户 (用于插值)
    long long* shares_flat,  // 展平的份额矩阵 [Rows=u2_len, Cols=vector_len]
    int vector_len,          // 秘密向量长度 = 2 + u1_len
    long long* ciphers_flat, // 展平的密文矩阵
    int data_len,            // 模型参数量
    long long* output_result // 输出结果
) {
    long long seed_global_0 = parse_long(seed_global_0_str);

    // =========================================================
    // 步骤 1: 批量恢复秘密向量 S
    // S = [Delta, Alpha, Beta_for_U1_0, Beta_for_U1_1, ...]
    // =========================================================
    std::vector<long long> reconstructed_secrets(vector_len);
    
    // 准备插值的 X 坐标 (x = client_id + 1)
    std::vector<int> x_coords;
    for(int i=0; i<u2_len; ++i) x_coords.push_back(u2_ids[i] + 1);

    printf("[ServerCore] Recovering Secret Vector (Len=%d) from %d clients...\n", vector_len, u2_len);

    // 对每一列(每一个秘密分量)进行插值
    for (int k = 0; k < vector_len; ++k) {
        std::vector<long long> y_coords;
        for (int i = 0; i < u2_len; ++i) {
            // Share Matrix Access: Row i, Col k
            long long share = shares_flat[i * (size_t)vector_len + k];
            y_coords.push_back(share);
        }
        reconstructed_secrets[k] = lagrange_interpolate_zero(x_coords, y_coords);
    }

    // =========================================================
    // 步骤 2: 解析秘密向量
    // =========================================================
    
    // 2.1 提取 Delta
    long long delta = reconstructed_secrets[0];
    printf("[ServerCore] RECONSTRUCTED DELTA: %lld\n", delta);

    // 2.2 提取 Alpha (可选，用于校验或直接使用)
    long long alpha_seed_rec = reconstructed_secrets[1];
    
    // 2.3 提取 Beta Seeds
    // 向量中剩下的部分是对应 u1_ids 顺序的 beta seeds
    // 我们需要构建一个查找表: ClientID -> BetaSeed
    std::map<int, long long> beta_map;
    for (int i = 0; i < u1_len; ++i) {
        // S 的结构是: [Delta, Alpha, Beta_0, Beta_1...]
        // 所以第 i 个用户的 Beta 在 index 2 + i
        if (2 + i < vector_len) {
            beta_map[u1_ids[i]] = reconstructed_secrets[2 + i];
        }
    }

    // =========================================================
    // 步骤 3: 准备消除噪声
    // Noise = (1 - Delta) * M + Sum(B_online)
    // =========================================================

    // 3.1 准备全局掩码 M
    // 使用恢复出的 alpha 种子来推导 M 的种子
    // 原逻辑: seed_M = (seed_g0 + seed_alpha)
    long seed_M = (seed_global_0 + alpha_seed_rec) & 0x7FFFFFFF;
    DeterministicRandom rng_M(seed_M);

    // 3.2 准备私有掩码 B (仅针对在线用户 U2)
    std::vector<DeterministicRandom> rng_B_list;
    for (int i = 0; i < u2_len; ++i) {
        int online_uid = u2_ids[i];
        if (beta_map.find(online_uid) != beta_map.end()) {
            long long s_b_long = beta_map[online_uid];
            // 还原为 int 种子
            rng_B_list.emplace_back((long)(s_b_long & 0x7FFFFFFF));
        } else {
            // 理论上不应发生，除非 U2 包含不在 U1 中的 ID
            printf("[ServerCore Error] Beta seed for online client %d missing!\n", online_uid);
        }
    }

    // 3.3 计算系数
    long long coeff_M = (1 - delta) % MOD;
    coeff_M = (coeff_M + MOD) % MOD;
    printf("[ServerCore] Coeff_M (1-Delta): %lld\n", coeff_M);

    // =========================================================
    // 步骤 4: 流式聚合与消去
    // =========================================================
    for (int k = 0; k < data_len; ++k) {
        // A. 生成噪声 N_k
        long long val_M = rng_M.next_mask_mod();
        long long term_M = MathUtils::safe_mod_mul(coeff_M, val_M, MOD);

        long long sum_B = 0;
        for (auto& rng : rng_B_list) sum_B += rng.next_mask_mod();
        sum_B %= MOD;
        
        long long noise_k = (term_M + sum_B) % MOD;

        // B. 累加密文 Sum(C_k)
        long long sum_cipher_k = 0;
        for(int i=0; i<u2_len; ++i) {
            long long val = ciphers_flat[i * (size_t)data_len + k];
            unsigned __int128 temp = (unsigned __int128)sum_cipher_k + val;
            sum_cipher_k = (long long)(temp % MOD);
        }

        // C. 消去: Result = Sum(C) - N
        long long res = (sum_cipher_k - noise_k) % MOD;
        res = (res + MOD) % MOD; // 保证正数

        output_result[k] = res;
    }
}

} // extern "C"