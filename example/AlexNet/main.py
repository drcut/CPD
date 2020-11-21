from __future__ import print_function

import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.distributed as dist
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms, models
from torch.distributed import broadcast
from CPDtorch.utils.dist_util import dist_init, sum_gradients, DistModule
import numpy as np
import torch.distributed as dist
import torch.nn as nn
import os
import math
from tqdm import tqdm
import random
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='PyTorch ImageNet Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--checkpoint-format', default='./checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--emulate-node', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing allreduce across workers; it multiplies '
                         'total batch size.')

parser.add_argument('--batch-size', type=int, default=128,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=256,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.0005,
                    help='weight decay')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--use-APS', action='store_true', default=False,
                    help='use APS algorithm')
parser.add_argument('--use-lars', action='store_true', default=False,
                    help='use APS algorithm')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument("--grad_exp", type=int)
parser.add_argument("--grad_man", type=int)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

allreduce_batch_size = args.batch_size * args.emulate_node

rank, world_size = dist_init()

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

cudnn.benchmark = True

class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

model = AlexNet().cuda()

# If set > 0, will resume training from a given checkpoint.
resume_from_epoch = 0

for try_epoch in range(args.epochs, 0, -1):
    if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
        resume_from_epoch = try_epoch
        break

# Horovod: print logs on the first worker.
verbose = 1 if rank == 0 else 0

# Horovod: limit # of CPU threads to be used per worker.
torch.set_num_threads(4)

# Data loading code
image_size = 256
input_size = 224
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean, std)

args.data = '/mnt/lustre/hanruobing/ppopp_test/images'
traindir = os.path.join(args.data, 'train')
train_dataset = datasets.ImageFolder(
    root=traindir,
    transform=transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
valdir = os.path.join(args.data, 'val')
val_dataset = datasets.ImageFolder(
    root=valdir,
    transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize,
    ]))
kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=world_size, rank=rank)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=allreduce_batch_size,
    sampler=train_sampler, **kwargs)

val_sampler = torch.utils.data.distributed.DistributedSampler(
    val_dataset, num_replicas=world_size, rank=rank)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size,
                                         sampler=val_sampler, **kwargs)


bn_params = [v for n, v in model.named_parameters() if 'bn' in n]
rest_params = [v for n, v in model.named_parameters() if not 'bn' in n]

optimizer = torch.optim.SGD([{'params': bn_params, 'weight_decay': 0},
                             {'params': rest_params, 'weight_decay': args.wd}],
                            3.2,
                            momentum=args.momentum,
                            weight_decay=args.wd,
                            nesterov=True)

model = DistModule(model)
if resume_from_epoch > 0:
    filepath = args.checkpoint_format.format(epoch=resume_from_epoch)
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

momentum_buffer = []
for master_p in model.parameters():
    momentum_buffer.append(torch.zeros_like(master_p))


def train(epoch):
    model.train()
    train_sampler.set_epoch(epoch)

    with tqdm(total=len(train_loader),
              desc='Train Epoch     #{}'.format(epoch),
              disable=not verbose) as t:
        train_loss = Metric('train_loss')
        train_accuracy = Metric('train_accuracy')
        for batch_idx, (data, target) in enumerate(train_loader):
            curr_lr = adjust_learning_rate(epoch, batch_idx)

            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            grad_buffer = []
            for param_g in model.parameters():
                grad_buffer.append([])
            # Split data into sub-batches of size batch_size
            for i in range(0, len(data), args.batch_size):
                data_batch = data[i:i + args.batch_size]
                target_batch = target[i:i + args.batch_size]
                output = model(data_batch)
                train_accuracy.update(accuracy(output, target_batch))
                loss = F.cross_entropy(
                    output, target_batch) / world_size
                reduced_loss = loss.data.clone()
                dist.all_reduce(reduced_loss)
                train_loss.update(float(reduced_loss.item()))
                # Average gradients among sub-batches
                loss.div_(math.ceil(float(len(data)) / args.batch_size))
                loss.backward()
            '''
                for idx, param in enumerate(model.parameters()):
                    if param.grad is not None:
                        grad_buffer[idx].append(
                            param.grad.detach().clone().data)
                model.zero_grad()
            for idx, param in enumerate(model.parameters()):
                if param.grad is not None:
                    # APS
                    # find maximum exponent
                    max_exp = -100
                    for val in grad_buffer[idx]:
                        t_exp = torch.log2(
                            torch.abs(val * args.emulate_node).max()).ceil().detach().cpu().numpy()
                        if t_exp > max_exp:
                            max_exp = t_exp
                    upper_bound = 2**(args.grad_exp - 1) - 1
                    shift_factor = upper_bound - max_exp
                    if max_exp == -100 or not args.use_APS:
                        shift_factor = 0
                    for grad in grad_buffer[idx]:
                        grad.data.copy_(float_quantize(
                            grad * (2**shift_factor), args.grad_exp, args.grad_man))
                    # as we use a single node to emulate multi-node, we should
                    # first accumulate gradients within a single node and then
                    # communicate them in the distributed system
                    res = torch.zeros_like(grad_buffer[idx][0])
                    for val in grad_buffer[idx]:
                        res = float_quantize(
                            res + val, args.grad_exp, args.grad_man)
                    param.grad.data.copy_(res.data / (2**shift_factor))
            '''
            sum_gradients(model, use_APS=args.use_APS,
                          grad_exp=args.grad_exp, grad_man=args.grad_man)

            # Gradient is applied across all ranks
            if args.use_lars:
                for idx, master_p in enumerate(model.parameters()):
                    if master_p.grad is not None:
                        local_lr = master_p.norm(2) /\
                            (master_p.grad.data.norm(2)
                             + args.wd * master_p.norm(2))
                        lars_coefficient = 0.001
                        local_lr = local_lr * lars_coefficient
                        momentum_buffer[idx] = args.momentum * momentum_buffer[idx].data \
                            + curr_lr \
                            * local_lr \
                            * (master_p.grad.data + args.wd * master_p.data)
                        update = momentum_buffer[idx]
                        master_p.data.copy_(master_p - update)
            else:
                optimizer.step()

            # optimizer.step()
            optimizer.zero_grad()

            t.set_postfix({'lr': curr_lr,
                           'loss': train_loss.avg,
                           'accuracy': 100. * train_accuracy.avg})
            t.update(1)


def validate(epoch):
    model.eval()
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')

    with tqdm(total=len(val_loader),
              desc='Validate Epoch  #{}'.format(epoch),
              disable=not verbose) as t:
        with torch.no_grad():
            for data, target in val_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                val_loss.update(F.cross_entropy(output, target).item())
                val_accuracy.update(accuracy(output, target))
                t.set_postfix({'loss': val_loss.avg,
                               'accuracy': 100. * val_accuracy.avg})
                t.update(1)
    print("Epoch:{} val loss:{} val accuracy:{}".format(
        epoch, val_loss.avg, val_accuracy.avg * 100.0))


def adjust_learning_rate(epoch, batch_idx):
    lr = 1.28
    if epoch <= args.warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_loader)
        final_lr = 1.28
        lr = 0.01 + (float(epoch - 1) / args.warmup_epochs) * (final_lr - 0.1)
    else:
        epoch += float(batch_idx + 1) / len(train_loader)
        lr = 1.28*((1 - (epoch-args.warmup_epochs)/(args.epochs-args.warmup_epochs))**2)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def save_checkpoint(epoch):
    if rank == 0:
        filepath = args.checkpoint_format.format(epoch=epoch)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }
        torch.save(state, filepath)


# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = 0
        self.n = 0

    def update(self, val):
        self.sum += val
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n

for epoch in range(resume_from_epoch + 1, args.epochs + 1):
    train(epoch)
    validate(epoch)
    save_checkpoint(epoch)
