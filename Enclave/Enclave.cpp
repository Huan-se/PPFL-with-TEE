/* Enclave/Enclave.cpp */
#include "Enclave_t.h"
#include <sgx_trts.h>
#include <stdio.h>  // 提供 vsnprintf, BUFSIZ
#include <stdlib.h> // 提供基础定义

// === [关键修复] 暴力声明丢失的 C 函数 ===
// SGX 头文件隐藏了 rand/srand，我们手动声明它们以满足 Eigen 的编译需求。
extern "C" {
    int rand(void);
    void srand(unsigned int seed);
}

// 确保 RAND_MAX 存在 (通常 SGX 环境下可能未定义)
#ifndef RAND_MAX
#define RAND_MAX 2147483647
#endif

// === [关键修复] 注入 std 命名空间 ===
namespace std {
    using ::rand;
    using ::srand;
}

// 现在引入 C++ 头文件
#include <vector>
#include <random>
#include <cmath>
#include <algorithm> 

// 引入 Eigen
#include <Eigen/Dense>

#define CHUNK_SIZE 4096

// === 修复 printf 实现 (void -> int) ===
int printf(const char *fmt, ...) {
    char buf[BUFSIZ] = { '\0' };
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, BUFSIZ, fmt, ap);
    va_end(ap);
    ocall_print_string(buf);
    return 0;
}

// 核心函数实现
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
    // 1. 初始化随机数引擎 (实际使用的随机源)
    std::mt19937 gen(static_cast<unsigned int>(seed));
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // 预分配栈内存
    Eigen::VectorXf proj_chunk(CHUNK_SIZE);

    // 2. 流式计算
    for (size_t k = 0; k < out_len; ++k) {
        
        float dot_product = 0.0f;
        
        // 3. 遍历切片
        for (size_t r = 0; r < ranges_len; r += 2) {
            int start_idx = ranges[r];
            int block_len = ranges[r+1];

            if (start_idx < 0 || start_idx + block_len > (int)model_len) {
                continue; 
            }

            // 4. 分块处理
            int offset = 0;
            while (offset < block_len) {
                int current_chunk_size = std::min((int)CHUNK_SIZE, block_len - offset);
                
                // A. 生成随机投影向量
                for (int i = 0; i < current_chunk_size; ++i) {
                    proj_chunk[i] = dist(gen);
                }

                // B. 映射输入数据
                int real_base_idx = start_idx + offset;
                Eigen::Map<Eigen::VectorXf> vec_new(w_new + real_base_idx, current_chunk_size);
                Eigen::Map<Eigen::VectorXf> vec_old(w_old + real_base_idx, current_chunk_size);
                
                // C. 截取投影向量
                auto vec_proj = proj_chunk.head(current_chunk_size);

                // D. 核心计算
                dot_product += (vec_new - vec_old).dot(vec_proj);

                offset += current_chunk_size;
            }
        }

        output[k] = dot_product;
    }
}