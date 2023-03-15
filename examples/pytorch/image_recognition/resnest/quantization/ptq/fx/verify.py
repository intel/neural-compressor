# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: Hang Zhang
# Email: zhanghang0704@gmail.com
# Copyright (c) 2020
#
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from __future__ import print_function
import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn

import PIL
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time

import resnest.torch as module
import inspect
import importlib

import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


class Options():
    def __init__(self):
        # data settings
        parser = argparse.ArgumentParser(description='Deep Encoding')
        parser.add_argument('--base-size', type=int, default=None,
                            help='base image size')
        parser.add_argument('--crop-size', type=int, default=224,
                            help='crop image size')
        # model params
        parser.add_argument('-a', '--model', type=str, default='densenet',
                            help='network model type (default: densenet)')
        # training hyper params
        parser.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                            help='batch size for training (default: 128)')
        parser.add_argument('-j', '--workers', type=int, default=32,
                            metavar='N', help='dataloader threads')
        # cuda, seed and logging
        parser.add_argument('--no-cuda', action='store_true',
                            default=False, help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        # checking point
        parser.add_argument('--resume', type=str, default=None,
                            help='put the path to resuming file if needed')
        parser.add_argument('--verify', type=str, default=None,
                            help='put the path to resuming file if needed')
        parser.add_argument("--tune", action='store_true',
                            help="run Neural Compressor to tune int8 acc.")
        parser.add_argument('data', metavar='DIR',
                            help='path to dataset')
        parser.add_argument('-i', '--iter', default=0, type=int, metavar='N',
                            help='number of total iterations to run')
        parser.add_argument('-w', '--warmup-iterations', default=5, type=int, metavar='N',
                            help='number of warmup iterations to run')
        parser.add_argument('--performance', dest='performance', action='store_true',
                            help='run benchmark')
        parser.add_argument('-r', "--accuracy", dest='accuracy', action='store_true',
                    help='For accuracy measurement only.')
        parser.add_argument("--tuned_checkpoint", default='./saved_results', type=str, metavar='PATH',
                            help='path to checkpoint tuned by Neural Compressor'
                                 ' (default: ./)')
        parser.add_argument('--int8', dest='int8', action='store_true',
                            help='run benchmark for int8')
        parser.add_argument('--warmup_iter', default=5, type=int,
                            help='For benchmark measurement only.')
        parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        return args


def main():
    # init the args
    args = Options().parse()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    # init dataloader
    interp = PIL.Image.BILINEAR if args.crop_size < 320 else PIL.Image.BICUBIC
    base_size = args.base_size if args.base_size is not None else int(1.0 * args.crop_size / 0.875)
    transform_val = transforms.Compose([
        ECenterCrop(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    valset = ImageNetDataset(args.data, transform=transform_val, train=False)
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True if args.cuda else False)

    # assert args.model in torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)
    functions = inspect.getmembers(module, inspect.isfunction)
    model_list = [f[0] for f in functions]
    assert args.model in model_list
    get_model = importlib.import_module('resnest.torch')
    net = getattr(get_model, args.model)
    from resnest.torch import resnest50
    model = resnest50(pretrained=True)
    # print(model)

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss()

    if args.cuda:
        model.cuda()
        # Please use CUDA_VISIBLE_DEVICES to control the number of gpus
        model = nn.DataParallel(model)

    # checkpoint
    if args.verify:
        if os.path.isfile(args.verify):
            print("=> loading checkpoint '{}'".format(args.verify))
            model.module.load_state_dict(torch.load(args.verify))
        else:
            raise RuntimeError("=> no verify checkpoint found at '{}'".
                               format(args.verify))
    elif args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.module.load_state_dict(checkpoint['state_dict'])
        else:
            raise RuntimeError("=> no resume checkpoint found at '{}'".
                               format(args.resume))

    model.eval()

    def eval_func(model):
        accu = validate(val_loader, model, criterion, args)
        return float(accu)

    if args.tune:
        from neural_compressor import PostTrainingQuantConfig
        from neural_compressor import quantization
        conf = PostTrainingQuantConfig()
        q_model = quantization.fit(model,
                                    conf,
                                    calib_dataloader=val_loader,
                                    eval_func=eval_func)
        q_model.save(args.tuned_checkpoint)
        return

    if args.performance or args.accuracy:
        model.eval()
        if args.int8:
            from neural_compressor.utils.pytorch import load
            new_model = load(os.path.abspath(os.path.expanduser(args.tuned_checkpoint)),
                             model,
                             dataloader=val_loader)
        else:
            new_model = model
        if args.performance:
            from neural_compressor.config import BenchmarkConfig
            from neural_compressor import benchmark
            b_conf = BenchmarkConfig(warmup=5,
                                     iteration=args.iter,
                                     cores_per_instance=4,
                                     num_of_instance=1)
            benchmark.fit(new_model, b_conf, b_dataloader=val_loader)
        if args.accuracy:
            validate(val_loader, new_model, criterion, args)
        return

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                            prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if i >= args.warmup_iter:
                start = time.time()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            if i >= args.warmup_iter:
                batch_time.update(time.time() - start)

            if i % args.print_freq == 0:
                progress.print(i)

        print('Batch size = %d' % args.batch_size)
        print('Accuracy: {top1:.5f} Accuracy@5 {top5:.5f}'
            .format(top1=(top1.avg / 100), top5=(top5.avg / 100)))

    return top1.avg


class ECenterCrop:
    """Crop the given PIL Image and resize it to desired size.
    Args:
        img (PIL Image): Image to be cropped. (0,0) denotes the top left corner of the image.
        output_size (sequence or int): (height, width) of the crop box. If int,
            it is used for both directions
    Returns:
        PIL Image: Cropped image.
    """
    def __init__(self, imgsize):
        self.imgsize = imgsize
        self.resize_method = transforms.Resize((imgsize, imgsize), interpolation=PIL.Image.BICUBIC)

    def __call__(self, img):
        image_width, image_height = img.size
        image_short = min(image_width, image_height)

        crop_size = float(self.imgsize) / (self.imgsize + 32) * image_short

        crop_height, crop_width = crop_size, crop_size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        img = img.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))
        return self.resize_method(img)


class ImageNetDataset(datasets.ImageFolder):
    # BASE_DIR = "ILSVRC2012"
    BASE_DIR = ""
    def __init__(self, root=os.path.expanduser('~/.encoding/data'), transform=None,
                 target_transform=None, train=True, **kwargs):
        split = 'train' if train == True else 'val'
        root = os.path.join(root, self.BASE_DIR, split)
        super(ImageNetDataset, self).__init__(root, transform, target_transform)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def save_profile_result(filename, table):
    import xlsxwriter
    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet()
    keys = ["Name", "Self CPU total %", "Self CPU total", "CPU total %", "CPU total",
            "CPU time avg", "Number of Calls"]
    for j in range(len(keys)):
        worksheet.write(0, j, keys[j])

    lines = table.split("\n")
    for i in range(3, len(lines)-4):
        words = lines[i].split(" ")
        j = 0
        for word in words:
            if not word == "":
                worksheet.write(i-2, j, word)
                j += 1
    workbook.close()


if __name__ == "__main__":
    main()
