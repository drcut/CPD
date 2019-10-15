#include <stdint.h>

__global__ void tvm_gemm(float *feature, float *kernel, float *gemm, int M,
                         int K, int N, int man_bits, int exp_bits);

__global__ void float_kernel_nearest(float *__restrict__ a, float *o, int size,
                                     int man_bits, int exp_bits);
