#include "quant_kernel.h"
//#include <ATen/ATen.h>
#include <torch/types.h>
#include <climits>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <tuple>

using namespace at;

torch::Tensor float_quantize_nearest_cuda(torch::Tensor a, int man_bits, int exp_bits) {
  // use external random number right now
  auto o = zeros_like(a);
  int size = a.numel();
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  float_kernel_nearest<<<blockNums, blockSize>>>(
      a.data<float>(), o.data<float>(), size, man_bits, exp_bits);
  return a;
  // return o;
}

void float_quantize_gemm_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor c, int M, int N, int K,
                              int man_bits, int exp_bits) {
  // use external random number right now
  dim3 grid_size;
  dim3 block_size;
  grid_size.x = ((M + 15) / 16);
  grid_size.y = ((N + 15) / 16);
  grid_size.z = 1;

  block_size.x = 8;
  block_size.y = 8;
  block_size.z = 1;
  tvm_gemm<<<grid_size, block_size>>>(a.data<float>(), b.data<float>(),
                                      c.data<float>(), M, K, N, man_bits,
                                      exp_bits);
  return;
}
