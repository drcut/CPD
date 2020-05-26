import torch
import torch.distributed as dist
from torch.nn import Module
from ..quant import float_quantize


class DistModule(Module):
    def __init__(self, module):
        super(DistModule, self).__init__()
        self.module = module
        broadcast_params(self.module)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def train(self, mode=True):
        super(DistModule, self).train(mode)
        self.module.train(mode)


def sum_gradients(model, use_APS=False, grad_exp=5, grad_man=2, use_kahan=False):
    if use_APS:
        world_size = dist.get_world_size()
        max_exp = []
        for idx, param in enumerate(model.parameters()):
            max_exp.append(torch.log2(
                torch.abs(param.grad.data * world_size).max()).ceil())
        max_exp = torch.Tensor(max_exp).cuda()
        dist.all_reduce(max_exp, op=dist.reduce_op.MAX)

        upper_bound = 2**(grad_exp-1) - 1
        shift_factor = [upper_bound - exp.detach().cpu().numpy()
                        for exp in max_exp]
        for idx, param in enumerate(model.parameters()):
            param.grad.copy_(float_quantize(
                param.grad.data*(2**shift_factor[idx]), grad_exp, grad_man))

        if not use_kahan:
            normal_sum_gradients(model, grad_exp, grad_man)
        else:
            kahan_sum_gradients(model, grad_exp, grad_man)

        for idx, param in enumerate(model.parameters()):
            param.grad.copy_(param.grad.data/(2**shift_factor[idx]))


def normal_sum_gradients(model, grad_exp=8, grad_man=23):
    if grad_exp == 8 and grad_man == 23:
        for _, param in model.named_parameters():
            if param.requires_grad:
                dist.all_reduce(param.grad.data)
        return
    for param in model.parameters():
        if param.requires_grad:
            gather_t = [torch.ones_like(param)
                        for _ in range(dist.get_world_size())]
            dist.all_gather(gather_t, param.grad.data)
            res = torch.zeros_like(param)
            for grad in gather_t:
                res = float_quantize(res+grad, grad_exp, grad_man)

            param.grad.data.copy_(res.data)


def kahan_sum_gradients(model, grad_exp=8, grad_man=23):
    if grad_exp == 8 and grad_man == 23:
        for _, param in model.named_parameters():
            if param.requires_grad:
                dist.all_reduce(param.grad.data)
        return
    for param in model.parameters():
        if param.requires_grad:
            gather_t = [torch.ones_like(param)
                        for _ in range(dist.get_world_size())]
            dist.all_gather(gather_t, param.grad.data)

            # Using Kahan Accumulation algorithm
            res = torch.zeros_like(param)
            c = torch.zeros_like(param)
            for grad in gather_t:
                y = float_quantize(grad - c, grad_exp, grad_man)
                t = float_quantize(res + y, grad_exp, grad_man)
                c = float_quantize(
                    float_quantize(t - res, grad_exp, grad_man) - y,
                    grad_exp, grad_man)
                res = t
            param.grad.data.copy_(res.data)


def broadcast_params(model):
    for p in model.state_dict().values():
        dist.broadcast(p, 0)
