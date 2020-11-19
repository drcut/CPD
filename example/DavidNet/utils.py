from CPDtorch.utils.dist_util import dist_init, sum_gradients, DistModule
from train_utils import AverageMeter, accuracy, DistributedGivenIterationSampler, DistributedSampler
from inspect import signature
from collections import namedtuple
import time
import torch
from torch import nn
import numpy as np
import torchvision
import torch
import torch.cuda
import torch.nn as nn
import torch.distributed as dist
import torch.optim
from torch.utils.data import distributed
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

torch.backends.cudnn.benchmark = True
np.random.seed(2)

#####################
# utils
#####################


class Timer():
    def __init__(self):
        self.times = [time.time()]
        self.total_time = 0.0

    def __call__(self, include_in_total=True):
        self.times.append(time.time())
        dt = self.times[-1] - self.times[-2]
        if include_in_total:
            self.total_time += dt
        return dt


def localtime(): return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


class TableLogger():
    def __init__(self, rank):
        self.rank = rank

    def append(self, output):
        if(self.rank != 0):
            return
        if not hasattr(self, 'keys'):
            self.keys = output.keys()
            print(*(f'{k:>12s}' for k in self.keys))
        filtered = [output[k] for k in self.keys]
        print(*(f'{v:12.4f}' if isinstance(v, np.float)
                else f'{v:12}' for v in filtered))

#####################
# data preprocessing
#####################


# equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_mean = (0.4914, 0.4822, 0.4465)
# equals np.std(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616)


def normalise(x, mean=cifar10_mean, std=cifar10_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean * 255
    x *= 1.0 / (255 * std)
    return x


def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border),
                      (border, border), (0, 0)], mode='reflect')


def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target])

#####################
# data augmentation
#####################


class Crop(namedtuple('Crop', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        return x[:, y0:y0 + self.h, x0:x0 + self.w]

    def options(self, x_shape):
        C, H, W = x_shape
        return {'x0': range(W + 1 - self.w), 'y0': range(H + 1 - self.h)}

    def output_shape(self, x_shape):
        C, H, W = x_shape
        return (C, self.h, self.w)


class FlipLR(namedtuple('FlipLR', ())):
    def __call__(self, x, choice):
        return x[:, :, ::-1].copy() if choice else x

    def options(self, x_shape):
        return {'choice': [True, False]}


class Cutout(namedtuple('Cutout', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        x = x.copy()
        x[:, y0:y0 + self.h, x0:x0 + self.w].fill(0.0)
        return x

    def options(self, x_shape):
        C, H, W = x_shape
        return {'x0': range(W + 1 - self.w), 'y0': range(H + 1 - self.h)}


class Transform():
    def __init__(self, dataset, transforms):
        self.dataset, self.transforms = dataset, transforms
        self.choices = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, labels = self.dataset[index]
        for choices, f in zip(self.choices, self.transforms):
            args = {k: v[index] for (k, v) in choices.items()}
            data = f(data, **args)
        return data, labels

    def set_random_choices(self):
        self.choices = []
        x_shape = self.dataset[0][0].shape
        N = len(self)
        for t in self.transforms:
            options = t.options(x_shape)
            x_shape = t.output_shape(x_shape) if hasattr(
                t, 'output_shape') else x_shape
            self.choices.append({k: np.random.choice(v, size=N)
                                 for (k, v) in options.items()})

#####################
# data loading
#####################


class Batches():
    def __init__(self, dataset, batch_size, shuffle, num_workers=0,
                 drop_last=False, sampler=None, half=None, dist_=None):
        self.dataset = dataset
        if half == 1:
            self.half = True
        else:
            self.half = False
        sampler = None
        if dist_ == 1:
            sampler = distributed.DistributedSampler(dataset)
            shuffle = False

        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers,
            pin_memory=True, shuffle=shuffle, sampler=sampler, drop_last=drop_last)

    def __iter__(self):
        if self.half == 1:
            return ({'input': x.cuda().half(), 'target': y.cuda()}
                    for (x, y) in self.dataloader)
        else:
            return ({'input': x.cuda(), 'target': y.cuda()}
                    for (x, y) in self.dataloader)

    def __len__(self):
        return len(self.dataloader)


#####################
# torch stuff
#####################

class Identity(nn.Module):
    def forward(self, x): return x


class Mul(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def __call__(self, x):
        return x * self.weight


class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), x.size(1))


class Add(nn.Module):
    def forward(self, x, y): return x + y


class Concat(nn.Module):
    def forward(self, *xs): return torch.cat(xs, 1)


class Correct(nn.Module):
    def forward(self, classifier, target):
        return classifier.max(dim=1)[1] == target


def batch_norm(num_channels, bn_bias_init=None, bn_bias_freeze=False,
               bn_weight_init=None, bn_weight_freeze=False, exp=8, man=23):
    m = nn.BatchNorm2d(num_channels)
    if bn_bias_init is not None:
        m.bias.data.fill_(bn_bias_init)
    if bn_bias_freeze:
        m.bias.requires_grad = False
    if bn_weight_init is not None:
        m.weight.data.fill_(bn_weight_init)
    if bn_weight_freeze:
        m.weight.requires_grad = False

    return m


def to_numpy(x):
    return x.detach().cpu().numpy()

#####################
# dict utils
#####################


union = lambda *dicts: {k: v for d in dicts for (k, v) in d.items()}


def path_iter(nested_dict, pfx=()):
    for name, val in nested_dict.items():
        if isinstance(val, dict):
            yield from path_iter(val, (*pfx, name))
        else:
            yield ((*pfx, name), val)

#####################
# graph building
#####################


sep = '_'
RelativePath = namedtuple('RelativePath', ('parts'))
rel_path = lambda *parts: RelativePath(parts)


def build_graph(net):
    net = dict(path_iter(net))
    default_inputs = [[('input',)]] + [[k] for k in net.keys()]

    def with_default_inputs(vals): return (
        val if isinstance(
            val,
            tuple) else (
            val,
            default_inputs[idx]) for idx,
        val in enumerate(vals))
    def parts(path, pfx): return tuple(pfx) + path.parts if isinstance(path,
                                                                       RelativePath) else (path,) if isinstance(path, str) else path
    return {sep.join((*pfx, name)): (val, [sep.join(parts(x, pfx)) for x in inputs]) for (
        *pfx, name), (val, inputs) in zip(net.keys(), with_default_inputs(net.values()))}


class TorchGraph(nn.Module):
    def __init__(self, net):
        self.graph = build_graph(net)
        super().__init__()
        for n, (v, _) in self.graph.items():
            setattr(self, n, v)

    def forward(self, inputs):
        self.cache = dict(inputs)
        for n, (_, i) in self.graph.items():
            self.cache[n] = getattr(self, n)(*[self.cache[x] for x in i])
        return self.cache

    def half(self):
        for module in self.children():
            if not isinstance(module, nn.BatchNorm2d):
                module.half()
        return self

#####################
# training utils
#####################


class PiecewiseLinear(namedtuple('PiecewiseLinear', ('knots', 'vals'))):
    def __call__(self, t):
        return np.interp([t], self.knots, self.vals)[0]


def trainable_params(model): return filter(
    lambda p: p.requires_grad, model.parameters())


def nesterov(params, momentum, weight_decay=None):
    return torch.optim.SGD(params, lr=0.0, momentum=momentum,
                           weight_decay=weight_decay, nesterov=True)


def concat(xs): return np.array(
    xs) if xs[0].shape is () else np.concatenate(xs)


def set_opt_params(optimizer, params, warm_up_iter):
    global global_step
    if global_step > warm_up_iter:
        for k, v in params.items():
            optimizer.param_groups[0][k] = v
    else:
        for k, v in params.items():
            optimizer.param_groups[0][k] = v * (global_step / warm_up_iter)
    return optimizer


def update(model, optimizer, dist_, grad_exp, grad_man, use_APS, loss_scale):
    assert model.training
    model.zero_grad()
    optimizer.zero_grad()
    if dist_ == 1:
        model.module.cache['loss'] = model.module.cache['loss'] * loss_scale
        model.module.cache['loss'].backward()

        sum_gradients(
            model,
            use_APS=use_APS,
            grad_exp=grad_exp,
            grad_man=grad_man)

    else:
        model.cache['loss'].backward()
    optimizer.step()


def collect(stats, output, dist_):
    for k, v in stats.items():
        if dist_ == 1:
            tmp = output[k].clone().detach()
            if tmp.type() == 'torch.cuda.BoolTensor':
                tmp = tmp.int()
            dist.all_reduce(tmp)
            v.append(to_numpy(tmp))
        else:
            v.append(to_numpy(output[k]))


global_step = 0


def train_epoch(model, batches, optimizer, lrs, stats, dist_,
                warm_up_iter, grad_exp, grad_man, use_APS, loss_scale):
    global global_step
    model.train(True)
    for lr, batch in zip(lrs, batches):
        collect(stats, model(batch), dist_)
        update(model,
               set_opt_params(optimizer,
                              {'lr': lr},
                              warm_up_iter),
               dist_,
               grad_exp,
               grad_man,
               use_APS,
               loss_scale)
        global_step += 1
    return stats


def test_epoch(model, batches, stats, dist_):
    model.train(False)
    for batch in batches:
        collect(stats, model(batch), dist_)
    return stats


def sum_(xs): return np.sum(concat(xs), dtype=np.float)


def train(model, lr_schedule, optimizer, train_set, test_set, batch_size=512,
          loggers=(), test_time_in_total=True, num_workers=0, drop_last=False, timer=None, args=None):

    t = timer or Timer()
    train_batches = Batches(train_set, batch_size, shuffle=True,
                            num_workers=num_workers, drop_last=drop_last, half=args.half, dist_=args.dist)
    test_batches = Batches(test_set, batch_size, shuffle=False,
                           num_workers=num_workers, half=args.half, dist_=args.dist)

    N_train, N_test = len(train_set), len(test_set)
    if drop_last:
        N_train -= (N_train % batch_size)

    for epoch in range(lr_schedule.knots[-1]):

        train_batches.dataset.set_random_choices()
        lrs = (
            lr_schedule(x) /
            batch_size for x in np.arange(
                epoch,
                epoch +
                1,
                1 /
                len(train_batches)))

        train_stats, train_time = train_epoch(model, train_batches, optimizer, lrs, {'loss': [], 'correct': [
        ]}, args.dist, args.warm_up_iter, args.grad_exp, args.grad_man, args.use_APS, args.loss_scale), t()
        test_stats, test_time = test_epoch(
            model, test_batches, {
                'loss': [], 'correct': []}, args.dist), t(test_time_in_total)

        record = optimizer.param_groups[0]['lr']
        summary = {
            'epoch': epoch + 1,
            'lr': record,
            'train time': train_time,
            'train loss': sum_(train_stats['loss']) / N_train,
            'train acc': sum_(train_stats['correct']) / N_train,
            'test time': test_time,
            'test loss': sum_(test_stats['loss']) / N_test,
            'test acc': sum_(test_stats['correct']) / N_test,
            'total time': t.total_time,
        }
        for logger in loggers:
            logger.append(summary)
    return summary
