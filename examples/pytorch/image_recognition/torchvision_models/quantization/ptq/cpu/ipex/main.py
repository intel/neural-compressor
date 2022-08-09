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
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models.quantization as quantize_models
import torchvision.models as models
from neural_compressor.adaptor.pytorch import get_torch_version, PyTorchVersionMode

try:
    try:
        import intel_pytorch_extension as ipex
        IPEX_110 = False
        IPEX_112 = False
    except:
        try:
            import intel_extension_for_pytorch as ipex
            from intel_extension_for_pytorch.quantization import prepare, convert
            from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig
            IPEX_110 = False
            IPEX_112 = True
        except:
            import intel_extension_for_pytorch as ipex
            import torch.fx.experimental.optimization as optimization
            IPEX_110 = True
            IPEX_112 = False
    TEST_IPEX = True
    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))
except:
    IPEX_110 = None
    TEST_IPEX = False
    model_names = sorted(name for name in quantize_models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(quantize_models.__dict__[name]))

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
hub_model_names = torch.hub.list('facebookresearch/WSL-Images')
model_names += hub_model_names

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--hub', action='store_true', default=False,
                    help='use model with torch hub')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--steps', default=-1, type=int,
                    help='steps for validation')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-t', '--tune', dest='tune', action='store_true',
                    help='tune best int8 model on calibration dataset')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--ppn', default=1, type=int,
                    help='number of processes on each node of distributed training')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('-i', "--iter", default=0, type=int,
                    help='For accuracy measurement only.')
parser.add_argument('-w', "--warmup_iter", default=5, type=int,
                    help='For benchmark measurement only.')
parser.add_argument('--benchmark', dest='benchmark', action='store_true',
                    help='run benchmark')
parser.add_argument('-r', "--accuracy_only", dest='accuracy_only', action='store_true',
                    help='For accuracy measurement only.')
parser.add_argument("--tuned_checkpoint", default='./saved_results', type=str, metavar='PATH',
                    help='path to checkpoint tuned by Neural Compressor (default: ./)')
parser.add_argument('--int8', dest='int8', action='store_true',
                    help='run benchmark')
parser.add_argument('--ipex', dest='ipex', action='store_true',
                    help='tuning or benchmark with Intel PyTorch Extension')

best_acc1 = 0


def main():
    args = parser.parse_args()
    print(args)

    if args.ipex:
        assert TEST_IPEX, 'Please import intel_pytorch_extension or intel_extension_for_pytorch according to version.'

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.ppn > 1 or args.multiprocessing_distributed

    if use_gpu:
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = args.ppn

    #ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    pytorch_version = get_torch_version()
    #args.gpu = gpu
    #affinity = subprocess.check_output("lscpu | grep 'NUMA node[0-9]' | awk '{ print $4 }' | awk -F',' '{ print $1 }'", shell=True)
    #os.environ['OMP_NUM_THREADS'] = '28'
    #os.environ['KMP_AFFINITY'] = 'proclist=[{}],granularity=thread,explicit'.format(affinity.splitlines()[gpu].decode('utf-8'))
    #print (os.environ['KMP_AFFINITY'])

    #if args.gpu is not None:
    #    print("Use GPU: {} for training".format(args.gpu))
    print("Use CPU: {} for training".format(gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    if args.hub:
        torch.set_flush_denormal(True)
        model = torch.hub.load('facebookresearch/WSL-Images', args.arch)
    else:
        # create model
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(args.arch))
            if args.ipex or pytorch_version >= PyTorchVersionMode.PT17.value:
                model = models.__dict__[args.arch](pretrained=True)
            else:
                model = quantize_models.__dict__[args.arch](pretrained=True, quantize=False)
        else:
            print("=> creating model '{}'".format(args.arch))
            if args.ipex:
                model = models.__dict__[args.arch]()
            else:
                model = quantize_models.__dict__[args.arch]()

    if args.ipex and not args.int8:
        model = model.to(memory_format=torch.channels_last)

    if not torch.cuda.is_available():
        print('using CPU...')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            #model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallelCPU(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    #cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)

    if args.tune:
        from neural_compressor.experimental import Quantization, common
        if args.ipex:
            quantizer = Quantization("./conf_ipex.yaml")
        else:
            model.eval()
            if pytorch_version < PyTorchVersionMode.PT17.value:
                model.fuse_model()
            quantizer = Quantization("./conf.yaml")
        quantizer.model = common.Model(model)
        q_model = quantizer.fit()
        q_model.save(args.tuned_checkpoint)
        return

    if args.benchmark or args.accuracy_only:
        model.eval()
        ipex_config_path = None
        if args.int8:
            if args.ipex:
                if not IPEX_110 and not IPEX_112:
                    # TODO: It will remove when IPEX spport to save script model.
                    model.to(ipex.DEVICE)
                    try:
                        new_model = torch.jit.script(model)
                    except:
                        new_model = torch.jit.trace(model, torch.randn(1, 3, 224, 224).to(ipex.DEVICE))
                else:
                    new_model = model
                ipex_config_path = os.path.join(os.path.expanduser(args.tuned_checkpoint),
                                                "best_configure.json")
            else:
                if pytorch_version < PyTorchVersionMode.PT17.value:
                    model.fuse_model()
                from neural_compressor.utils.pytorch import load
                new_model = load(
                    os.path.abspath(os.path.expanduser(args.tuned_checkpoint)), model)
        else:
            if args.ipex:
                if not IPEX_110 and not IPEX_112:
                    # TODO: It will remove when IPEX spport to save script model.
                    model.to(ipex.DEVICE)
                    try:
                        new_model = torch.jit.script(model)
                    except:
                        new_model = torch.jit.trace(model, torch.randn(1, 3, 224, 224).to(ipex.DEVICE))
                else:
                    model = ipex.optimize(model, dtype=torch.float32, inplace=True)
                    x = torch.randn(args.batch_size, 3, 224, 224).contiguous(memory_format=torch.channels_last)
                    with torch.no_grad():
                        model = torch.jit.trace(model, x).eval()
                    model = torch.jit.freeze(model)
                    new_model = model
            else:
                model.fuse_model()
                new_model = model
        validate(val_loader, new_model, criterion, args, ipex_config_path)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)


def validate(val_loader, model, criterion, args, ipex_config_path=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    if args.ipex:
        if not IPEX_110 and not IPEX_112:
            conf = (
                ipex.AmpConf(torch.int8, configure_file=ipex_config_path)
                if ipex_config_path is not None
                else ipex.AmpConf(None)
                )
        if IPEX_110:
            if ipex_config_path is not None:
                conf = ipex.quantization.QuantConf(configure_file=ipex_config_path)
                model = optimization.fuse(model, inplace=True)
                for idx, (input, label) in enumerate(val_loader):
                    x = input.contiguous(memory_format=torch.channels_last)
                    break
                model = ipex.quantization.convert(model, conf, x)
            else:
                model = model
        if IPEX_112:
            if ipex_config_path is not None:
                x = torch.randn(args.batch_size, 3, 224, 224).contiguous(memory_format=torch.channels_last) 
                qconfig = QConfig(
                        activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8),
                        weight= PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))
                prepared_model = ipex.quantization.prepare(model, qconfig, x, inplace=True)
                prepared_model.load_qconf_summary(qconf_summary=ipex_config_path)
                model = ipex.quantization.convert(prepared_model)
                model = torch.jit.trace(model, x)
                model = torch.jit.freeze(model.eval())
                y = model(x)
                y = model(x)
                print("running int8 model\n")
            else:
                model = model
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if i >= args.warmup_iter:
                start = time.time()
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            if args.ipex:
                if not IPEX_110 and not IPEX_112:
                    with ipex.AutoMixPrecision(conf, running_mode='inference'):
                        output = model(input.to(ipex.DEVICE))
                    target = target.to(ipex.DEVICE)
                else:
                    output = model(input)
            else:
                output = model(input)
           
           # measure elapsed time
            if i >= args.warmup_iter:
                batch_time.update(time.time() - start)

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))


            if i % args.print_freq == 0:
                progress.print(i)

            if args.iter > 0 and i >= (args.warmup_iter + args.iter - 1):
                break

        print('Batch size = %d' % args.batch_size)
        if args.batch_size == 1:
            print('Latency: %.3f ms' % (batch_time.avg * 1000))
        print('Throughput: %.3f images/sec' % (args.batch_size / batch_time.avg))

        # TODO: this should also be done with the ProgressMeter
        print('Accuracy: {top1:.5f} Accuracy@5 {top5:.5f}'
              .format(top1=(top1.avg / 100), top5=(top5.avg / 100)))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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


if __name__ == '__main__':
    main()
