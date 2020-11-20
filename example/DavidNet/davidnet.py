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
from torch.utils.data import DataLoader

from utils import *


# Network definition
def conv_bn(c_in, c_out, bn_weight_init=1.0, **kw):
    return {
        'conv': nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False),
        'bn': batch_norm(c_out, bn_weight_init=bn_weight_init, **kw),
        'relu': nn.ReLU(True)
    }


def residual(c, **kw):
    return {
        'in': Identity(),
        'res1': conv_bn(c, c, **kw),
        'res2': conv_bn(c, c, **kw),
        'add': (Add(), [rel_path('in'), rel_path('res2', 'relu')]),
    }


def basic_net(channels, weight, pool, **kw):
    return {
        'prep': conv_bn(3, channels['prep'], **kw),
        'layer1': dict(conv_bn(channels['prep'], channels['layer1'], **kw), pool=pool),
        'layer2': dict(conv_bn(channels['layer1'], channels['layer2'], **kw), pool=pool),
        'layer3': dict(conv_bn(channels['layer2'], channels['layer3'], **kw), pool=pool),
        'classifier': {
            'pool': nn.MaxPool2d(4),
            'flatten': Flatten(),
            'linear': nn.Linear(channels['layer3'], 10, bias=False),
            'logits': Mul(weight),
        }
    }


def net(channels=None, weight=0.125, pool=nn.MaxPool2d(2),
        extra_layers=(), res_layers=('layer1', 'layer3'), **kw):
    channels = channels or {
        'prep': 64,
        'layer1': 128,
        'layer2': 256,
        'layer3': 512}
    n = basic_net(channels, weight, pool, **kw)
    for layer in res_layers:
        n[layer]['residual'] = residual(channels[layer], **kw)
    for layer in extra_layers:
        n[layer]['extra'] = conv_bn(channels[layer], channels[layer], **kw)
    return n


losses = {
    'loss': (nn.CrossEntropyLoss(size_average=False), [('classifier', 'logits'), ('target',)]),
    'correct': (Correct(), [('classifier', 'logits'), ('target',)]),
}
