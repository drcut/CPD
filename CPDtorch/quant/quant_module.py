import torch
import torch.nn as nn
import torch.nn.init as init
import math
from torch.autograd import Function
from torch.nn.parameter import Parameter
from .quant_function import *
import numpy as np

__all__ = ['Quantizer', 'Quant_Linear', 'Quant_Conv']


class Quantizer(nn.Module):
    def __init__(self, forward_exp=8, forward_man=23, backward_exp=8, backward_man=23):
        super(Quantizer, self).__init__()
        self.quantize = quantizer(forward_exp, forward_man,
                                  backward_exp, backward_man)

    def forward(self, x):
        return self.quantize(x)


class Quant_LinearFunction(Function):

    @staticmethod
    def forward(ctx, input, weight, bias=None, exp=8, man=23):
        ctx.save_for_backward(input, weight, bias)
        ctx.exp = exp
        ctx.man = man
        output = quant_gemm(input, weight.t(), man=man, exp=exp)
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            # grad_input = grad_output.mm(weight)
            grad_input = quant_gemm(
                grad_output, weight, man=ctx.man, exp=ctx.exp)
        if ctx.needs_input_grad[1]:
            # grad_weight = grad_output.t().mm(input)
            grad_weight = quant_gemm(
                grad_output.t(), input, man=ctx.man, exp=ctx.exp)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = float_quantize(
                grad_output.sum(0).squeeze(0), ctx.exp, ctx.man)

        return grad_input, grad_weight, grad_bias, None, None


class Quant_Linear(nn.Module):

    def __init__(self, in_features, out_features, bias=True, exp=8, man=23):
        super(Quant_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.exp = exp
        self.man = man
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return Quant_LinearFunction.apply(input, self.weight, self.bias, self.exp, self.man)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Quant_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, exp=8, man=23):
        super(Quant_Conv, self).__init__()
        self.weight = Parameter(torch.Tensor(
            out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.reset_parameters()
        self.man = man
        self.exp = exp

    def reset_parameters(self):
        n = self.in_channels
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        if self.bias is not None:
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        batch, in_channel, in_height, in_width = input.shape
        num_filter, channel, kernel_h, kernel_w = self.weight.shape

        out_height = (in_height - kernel_h + 2 *
                      self.padding) // self.stride + 1
        out_width = (in_width - kernel_w + 2 * self.padding) // self.stride + 1

        def tmp_matmul(a, b, bias, exp, man):
            assert len(a.shape) == 3
            assert len(b.shape) == 2
            # will be tranpose later
            assert a.shape[2] == b.shape[1]
            batch, m, k = a.shape
            n = b.shape[0]
            a = a.contiguous()
            b = b.contiguous()
            a = a.view(batch*m, k)
            return Quant_LinearFunction.apply(a, b, bias, exp, man).view(batch, m, n)

        inp_unf = torch.nn.functional.unfold(
            input, self.kernel_size, stride=self.stride, padding=self.padding).transpose(1, 2)
        out_unf = tmp_matmul(inp_unf, self.weight.view(self.weight.size(0), -1),
                             self.bias, self.exp, self.man).transpose(1, 2)
        return out_unf.view(batch, num_filter, out_height, out_width)
