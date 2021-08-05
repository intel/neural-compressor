import argparse
import os
import random
import shutil
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models.quantization as models
from collections import OrderedDict

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-t', '--tune', dest='tune', action='store_true',
                    help='tune best int8 model on calibration dataset')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument("--config", default=None, help="tuning config")
parser.add_argument('-i', "--iter", default=0, type=int,
                    help='For accuracy measurement only.')
parser.add_argument("--tuned_checkpoint", default='./saved_results', type=str, metavar='PATH',
                    help='path to checkpoint tuned by Low Precision Optimization Tool (default: ./)')
parser.add_argument('--int8', dest='int8', action='store_true',
                    help='run benchmark')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True, quantize=False)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if args.tune:
        model.fuse_model()
        from lpot.experimental import Quantization, common, Pruning, Component
        from lpot.experimental.scheduler import Scheduler
        quantizer = Quantization('./qat_conf.yaml')
        prune = Pruning('./prune_conf.yaml')
        scheduler = Scheduler()
        scheduler.model = common.Model(model)
        combination = scheduler.combine(prune, quantizer)
        print(combination)
        scheduler.append(combination)
        opt_model = scheduler()
        opt_model.save(args.tuned_checkpoint)
        return

if __name__ == '__main__':
    main()
