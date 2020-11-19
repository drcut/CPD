from CPDtorch.utils.dist_util import dist_init, sum_gradients, DistModule
from train_utils import AverageMeter, accuracy, DistributedGivenIterationSampler, DistributedSampler
from davidnet import *
from utils import *
import argparse
import time
import math
import torch
import torch.backends.cudnn

parser = argparse.ArgumentParser()
parser.add_argument('--dist', default=1, type=int)
parser.add_argument('--epoch', default=24, type=int)
parser.add_argument('--warm_up_epoch', default=5, type=int)
parser.add_argument('-b', '--batch_size', default=512, type=int)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--workers', default=4)
parser.add_argument('--half', default=0, type=int)
parser.add_argument('--lr_scale', default=1.0, type=float)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--grad_exp', default=8, type=int)
parser.add_argument('--grad_man', default=23, type=int)
parser.add_argument('--use_APS', action='store_true')
parser.add_argument('--loss_scale', default=1, type=int)

args = parser.parse_args()

rank = 0
world_size = 1
dataset_len = None

torch.backends.cudnn.benchmark = True
torch.cuda.manual_seed_all(args.seed)
torch.manual_seed(args.seed)


class TSVLogger():
    def __init__(self):
        self.log = ['epoch\thours\ttop1Accuracy']

    def append(self, output):
        # output['test acc']*100
        epoch, hours, acc = output['epoch'], output['total time'] / 3600, 0
        self.log.append(f'{epoch}\t{hours:.8f}\t{acc:.2f}')

    def __str__(self):
        return '\n'.join(self.log)


def main():
    global args, rank, world_size
    if args.dist == 1:
        rank, world_size = dist_init()
    else:
        rank = 0
        world_size = 1

    DATA_DIR = './data'

    train_set_raw = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True)
    test_set_raw = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True)

    lr_schedule = PiecewiseLinear([0, 5, 24], [0, 0.4 * args.lr_scale, 0])
    train_transforms = [Crop(32, 32), FlipLR(), Cutout(8, 8)]

    model = TorchGraph(union(net(), losses)).cuda()
    if args.half == 1:
        model = model.half()
    if args.dist == 1:
        model = DistModule(model)
    opt = torch.optim.SGD(
        model.parameters(),
        lr=0.0,
        momentum=args.momentum,
        weight_decay=5e-4 *
        args.batch_size,
        nesterov=True)

    t = Timer()

    train_set = list(
        zip(transpose(normalise(pad(train_set_raw.data, 4))), train_set_raw.targets))
    test_set = list(
        zip(transpose(normalise(test_set_raw.data)), test_set_raw.targets))
    dataset_len = len(train_set)
    args.warm_up_iter = math.ceil(
        dataset_len * args.warm_up_epoch / (world_size * args.batch_size))

    TSV = TSVLogger()
    train(model, lr_schedule, opt, Transform(train_set, train_transforms), test_set, args=args,
          batch_size=args.batch_size, num_workers=args.workers, loggers=(TableLogger(rank), TSV), timer=t, test_time_in_total=False, drop_last=True)


main()
