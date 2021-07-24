import argparse
import os
import random
import shutil
import time
import warnings
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
use_gpu = False
if use_gpu:
    import torch.backends.cudnn as cudnn
#import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models.quantization as quantize_models

import subprocess

model_names = sorted(name for name in quantize_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(quantize_models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-t', '--tune', dest='tune', action='store_true',
                    help='tune best int8 model on calibration dataset')
parser.add_argument('--prune', dest='prune', action='store_true',
                    help='prune sparse model on calibration dataset')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument("--tuned_checkpoint", default='./saved_results', type=str, metavar='PATH',
                    help='path to checkpoint tuned by Low Precision Optimization Tool (default: ./)')

def main():
    args = parser.parse_args()
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = quantize_models.__dict__[args.arch](pretrained=True, quantize=False)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = quantize_models.__dict__[args.arch]()

    print('using CPU...')

    if args.tune and args.prune:
        from lpot.experimental.scheduler import Scheduler
        from lpot.experimental import Quantization, Pruning, common
        from lpot.adaptor.tf_utils.util import is_saved_model_format, is_ckpt_format
        prune = Pruning('./prune_conf.yaml')
        quantizer = Quantization('./ptq_conf.yaml')
        scheduler = Scheduler()
        scheduler.model = common.Model(model)
        scheduler.append(prune)
        scheduler.append(quantizer)
        opt_model = scheduler()
        opt_model.save(args.tuned_checkpoint)
        return

    elif args.tune:
        from lpot.experimental import Quantization, common
        model.eval()
        model.fuse_model()
        quantizer = Quantization("./ptq_conf.yaml")
        quantizer.model = common.Model(model)
        q_model = quantizer()
        q_model.save(args.tuned_checkpoint)
        return

    elif args.prune:
        from lpot.experimental import Pruning, common
        prune = Pruning('./prune_conf.yaml')

        prune.model = common.Model(model)
        p_model = prune()
        p_model.save(args.tuned_checkpoint)
        return

if __name__ == '__main__':
    main()
