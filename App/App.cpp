/* App/App.cpp */
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <sgx_urts.h> // SGX 不受信任运行时库
#include "App.h"      // 声明 tee_init 等函数
#include "Enclave_u.h" // 自动生成的 Enclave 代理头文件

// 全局变量保存 Enclave ID
sgx_enclave_id_t global_eid = 0;

// OCALL 实现：Enclave 内部调用打印时会回到这里
void ocall_print_string(const char *str) {
    printf("%s", str);
}

// 导出 C 接口给 Python ctypes 使用
extern "C" {

    // 1. 初始化 Enclave
    int tee_init(const char* enclave_filename) {
        sgx_status_t ret = sgx_create_enclave(
            enclave_filename, 
            SGX_DEBUG_FLAG, 
            NULL, 
            NULL, 
            &global_eid, 
            NULL
        );
        
        if (ret != SGX_SUCCESS) {
            printf("[App] Error: Failed to create enclave. Status: 0x%x\n", ret);
            return -1;
        }
        // printf("[App] Enclave initialized. ID: %lu\n", global_eid);
        return 0;
    }

    // 2. 销毁 Enclave
    void tee_destroy() {
        if (global_eid != 0) {
            sgx_destroy_enclave(global_eid);
            global_eid = 0;
            // printf("[App] Enclave destroyed.\n");
        }
    }

    // 3. 核心业务：安全聚合
    int tee_secure_aggregation(
        long seed,
        float* w_new,
        float* w_old,
        int model_len,
        int* ranges,
        int ranges_len,
        float* output,
        int out_len
    ) {
        if (global_eid == 0) {
            printf("[App] Error: Enclave not initialized!\n");
            return -1;
        }

        sgx_status_t status = ecall_secure_aggregation_phase(
            global_eid,
            seed,
            w_new,
            w_old,
            (size_t)model_len,
            ranges,
            (size_t)ranges_len,
            output,
            (size_t)out_len
        );

        if (status != SGX_SUCCESS) {
            printf("[App] Error: Ecall failed. Status: 0x%x\n", status);
            return -1;
        }
        return 0;
    }
}