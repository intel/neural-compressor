import argparse
import os
import random
import shutil
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import horovod.torch as hvd

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-t', '--topology', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet18)')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--prune', dest='prune', action='store_true',
                    help='prune sparse model on calibration dataset')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument("--config", default=None, help="tuning config")
parser.add_argument("--output-model", default=None, help="output path", type=str)

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
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
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--iteration', default=-1, type=int,
                    help='num of iteration of each epoch.')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--num-per-node', default=1, type=int,
                    help='number per node for distributed training')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--keep-batch-size', dest='keep_batch_size',
                    action='store_true',
                    help='keep the batch size rather than scale lr')

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    args.distributed = args.world_size > 1 or args.num_per_node > 1
    main_worker(args)


def main_worker(args):
    global best_acc1

    if args.distributed:
        hvd.init()
        print(hvd.size(), args.world_size, args.num_per_node)
        assert(hvd.size() == args.world_size * args.num_per_node)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.topology))
        # model = models.__dict__[args.topology](pretrained=True, quantize=False)
        model = models.__dict__[args.topology](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.topology))
        model = models.__dict__[args.topology]()

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
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=False)
        if args.keep_batch_size:
            assert(args.batch_size % hvd.size() == 0)
            args.batch_size = args.batch_size // hvd.size()
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    criterion = nn.CrossEntropyLoss()
    if args.distributed:
        # Horovod: scale learning rate by the number of rank if not keep the batch size.
        lr_scaler = 1 if args.keep_batch_size else hvd.size()
        optimizer = optim.SGD(model.parameters(),
                              lr=(args.lr * lr_scaler),
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
        # Horovod: wrap optimizer with DistributedOptimizer.
        optimizer = hvd.DistributedOptimizer(
            optimizer, named_parameters=model.named_parameters(),
            )
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    if args.prune:
        from neural_compressor.experimental import Pruning, common
        prune = Pruning(args.config)

        def train_func(model):
            # To add a learning rate scheduler, here redefine a training func

            global best_acc1
            # prune.on_train_begin()
            if args.distributed:
                # Horovod: broadcast parameters & optimizer state.
                hvd.broadcast_parameters(model.state_dict(), root_rank=0)
                hvd.broadcast_optimizer_state(optimizer, root_rank=0)

            for epoch in range(args.start_epoch, args.epochs):
                if args.distributed:
                    train_sampler.set_epoch(epoch)
                adjust_learning_rate(optimizer, epoch, args)

                prune.on_epoch_begin(epoch)
                if args.distributed:
                    train_loader.sampler.set_epoch(epoch)

                # train for one epoch
                train(train_loader, model, criterion, optimizer, epoch, args, prune)
                prune.on_epoch_end()

                # evaluate on validation set
                acc1, acc5 = validate(val_loader, model, criterion, args)

                # remember best acc@1 and save checkpoint
                is_best = acc1 > best_acc1
                best_acc1 = max(acc1, best_acc1)

                if not args.distributed or (args.distributed 
                    and hvd.rank() == 0):
                    print("=> save checkpoint")
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'topology': args.topology,
                        'state_dict': model.state_dict(),
                        'best_acc1': best_acc1,
                        'optimizer': optimizer.state_dict(),
                    }, is_best)
            return model

        prune.model = common.Model(model)
        prune._train_dataloader = train_loader
        prune._eval_dataloader = val_loader 
        prune._pruning_func = train_func 
        model = prune.fit()
        model.save(args.output_model)
        return


def train(train_loader, model, criterion, optimizer, epoch, args, op):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(args.iteration, batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        op.on_step_begin(i)
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

        op.on_before_optimizer_step()

        optimizer.step()

        op.on_step_end()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        if (i % args.print_freq == 0
            and (not args.distributed
                or (args.distributed and hvd.rank() == 0
            ))):
            progress.print(i)

        if args.iteration > 0 and i > args.iteration:
            break


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
            # compute output
            output = model(input)
            loss = criterion(output, target)
            
            if getattr(val_loader, 'distributed', False) or \
                 isinstance(val_loader.sampler, \
                 torch.utils.data.distributed.DistributedSampler):
                 hvd.init()
                 output = hvd.allgather(output)
                 target = hvd.allgather(target)
            
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            if (i % args.print_freq == 0 
                and (not args.distributed 
                    or (args.distributed and hvd.rank() == 0
                ))):
                progress.print(i)

        if not args.distributed or (args.distributed and hvd.rank() == 0):
            # TODO: this should also be done with the ProgressMeter
            print('Accuracy: {top1:.5f} Accuracy@5 {top5:.5f}'
                  .format(top1=(top1.avg / 100), top5=(top5.avg / 100)))

    return top1.avg, losses.avg


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

