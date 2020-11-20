import os
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
    else:
        if use_kahan:
            kahan_sum_gradients(model, grad_exp, grad_man)
        else:
            normal_sum_gradients(model, grad_exp, grad_man)
        return


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

def dist_init():
    if not 'SLURM_NODELIST' in os.environ.keys():
        raise NotImplementedError('We only support SLURM for distributed system')
    node_list = os.environ['SLURM_NODELIST']
    if '[' in node_list:
        beg = node_list.find('[')
        pos1 = node_list.find('-', beg)
        if pos1 < 0:
            pos1 = 1000
        pos2 = node_list.find(',', beg)
        if pos2 < 0:
            pos2 = 1000
        node_list = node_list[:min(pos1,pos2)].replace('[', '')
    host_name = node_list[8:].replace('-', '.')
    if 'SLURM_PROCID' in os.environ:
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
    elif 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
        proc_id = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        ntasks = int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
    else:
        raise NotImplementedError("only support openmpi/slurm multi-card training!")

    print("rank {0} of {1}, host {2}".format(
                  proc_id, ntasks, host_name))
    os.environ['MASTER_PORT'] = str(12345)
    os.environ['MASTER_ADDR'] = host_name
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
        
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size
