/* App/App.h */
#ifndef _APP_H_
#define _APP_H_

#if defined(__cplusplus)
extern "C" {
#endif

int tee_init(const char* enclave_filename);
void tee_destroy();
int tee_secure_aggregation(
    long seed, 
    float* w_new, 
    float* w_old, 
    int model_len, 
    int* ranges, 
    int ranges_len, 
    float* output, 
    int out_len
);

#if defined(__cplusplus)
}
#endif

#endif /* _APP_H_ */