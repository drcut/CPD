//#include <ATen/ATen.h>
#include <torch/extension.h>

#include <tuple>

using namespace at;

/**
 * quantize a Floattorch::Tensor into a low bit-width floating point torch::Tensor
 * with [man_bits] mantissa bits and [exp_bits] exponent bits.
 * Does not handle NaN, Inf, and denormal.
 * Nearest Rounding.
 **/
torch::Tensor float_quantize_nearest_cuda(torch::Tensor a, int man_bits, int exp_bits);

void float_quantize_gemm_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor c, int M, int N, int K, int man_bits, int exp_bits);
