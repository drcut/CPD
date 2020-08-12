#include "quant_cuda.h"
#include <torch/extension.h>
#include <torch/types.h>
#include <tuple>

using namespace at;

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  AT_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

torch::Tensor float_quantize_nearest(torch::Tensor a, int man_bits, int exp_bits) {
  CHECK_INPUT(a);
  return float_quantize_nearest_cuda(a, man_bits, exp_bits);
}

void float_quantize_gemm(torch::Tensor a, torch::Tensor b, torch::Tensor c, 
                         int M, int N, int K, int man_bits, int exp_bits) {
  CHECK_INPUT(a);
  CHECK_INPUT(b);
  CHECK_INPUT(c);
  float_quantize_gemm_cuda(a, b, c, M, N, K, man_bits, exp_bits);
  return;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("float_quantize_nearest", &float_quantize_nearest,
        "Low-Bitwidth Floating Point Number Nearest Neighbor Quantization "
        "(CUDA)");
  m.def("float_quantize_gemm", &float_quantize_gemm,
        "Low-Bitwidth Floating Point Number GEMM Quantization (CUDA)");
}

