import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.cpp_extension import load
import os
current_path = os.path.dirname(os.path.realpath(__file__))

if torch.cuda.is_available():
    quant_cuda = load(
        name='quant_cuda',
        sources=[
            os.path.join(current_path, "quant_cuda/quant_cuda.cpp"),
            os.path.join(current_path, "quant_cuda/float_kernel.cu"),
            os.path.join(current_path, "quant_cuda/quant.cu"),
        ],
    )
else:
    quant_cuda = None

__all__ = ['float_quantize', "quantizer", "quant_gemm"]


def get_module(x):
    if x.is_cuda:
        quant_module = quant_cuda
    else:
        raise NotImplementedError(
            "Currently, we do not support customized precision for CPU")
    return quant_module


def quantizer(forward_exp=8, forward_man=23, backward_exp=8, backward_man=23):

    class Rounding(torch.autograd.Function):
        @staticmethod
        def forward(self, x):
            if forward_exp == 8 and forward_man == 23:
                return x
            quant_module = get_module(x)
            out = quant_module.float_quantize_nearest(
                x.contiguous(), forward_man, forward_exp)
            return out

        @staticmethod
        def backward(self, grad_output):
            if self.needs_input_grad[0]:
                if backward_exp == 8 and backward_man == 23:
                    return grad_output
                quant_module = get_module(grad_output)
                grad_input = quant_module.float_quantize_nearest(
                    grad_output.contiguous(), backward_man, backward_exp)
            else:
                grad_input = None
            return grad_input

    return Rounding.apply


def float_quantize(x, exp, man):
    """
    Quantize a single precision Floating Point into low-precision Floating Point

    Args:
        - :attr: `x` (torch.Tensor) : the single precision number(torch.Tensor) to be quantized
        - :attr: `exp` (int) : number of bits allocated for exponent
        - :attr: `man` (int) : number of bits allocated for mantissa, not counting the virtual bit

    Returns:
        - a quantized low-precision floating point number (torch.Tensor)
    """
    assert isinstance(
        x, torch.Tensor), "x is not a single precision Floating Point Tensor"
    quant_module = get_module(x)
    return quant_module.float_quantize_nearest(x.contiguous(), man, exp)


def quant_gemm(a, b, man=23, exp=8):
    """
    Quantize GEMM with customized precision as accumulator

    Args:
        - :attr: `a` (torch.Tensor) : the input of GEMM, with shape:(M, K)
        - :attr: `b` (torch.Tensor) : the input of GEMM, with shape:(K, N)
        - :attr: `exp` (int) : number of bits allocated for exponent
        - :attr: `man` (int) : number of bits allocated for mantissa, not counting the virtual bit

    Returns:
        - the result of GEMM (torch.Tensor)
    """
    assert len(a.shape) == 2
    assert len(b.shape) == 2
    assert a.shape[1] == b.shape[0]
    quant_module = get_module(a)
    c = torch.zeros(a.shape[0], b.shape[1]).cuda()
    quant_module.float_quantize_gemm(a.contiguous(), b.contiguous(), c.contiguous(),
                                     a.shape[0], b.shape[1], a.shape[1], man, exp)
    return c
