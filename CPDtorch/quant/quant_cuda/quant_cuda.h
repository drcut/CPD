#include <ATen/ATen.h>
#include <tuple>

using namespace at;

/**
 * quantize a FloatTensor into a low bit-width floating point Tensor
 * with [man_bits] mantissa bits and [exp_bits] exponent bits.
 * Does not handle NaN, Inf, and denormal.
 * Nearest Rounding.
 **/
Tensor float_quantize_nearest_cuda(Tensor a, int man_bits, int exp_bits);

void float_quantize_gemm_cuda(Tensor a, Tensor b, Tensor c, int M, int N, int K, int man_bits, int exp_bits);
