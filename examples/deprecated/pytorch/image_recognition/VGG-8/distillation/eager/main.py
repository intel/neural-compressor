import os
import time
import shutil
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.vgg as vgg
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from plain_cnn_cifar import ConvNetMaker, plane_cifar100_book

# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

parser = argparse.ArgumentParser(description='PyTorch CNN or VGG Training')
parser.add_argument('--dataset', default='cifar100', type=str,
                    help='dataset cifar100')
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.02, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='CNN-2', type=str,
                    help='name of experiment')
parser.add_argument('--student_type', default='CNN-2', type=str,
                    help='type of student model (CNN-2 [default] or VGG-8)')
parser.add_argument('--teacher_type', default='CNN-10', type=str,
                    help='type of teacher model (CNN-10 [default] or VGG-13)')
parser.add_argument('--teacher_model', default='runs/CNN-10/model_best.pth.tar', type=str,
                    help='path of teacher model')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')

parser.add_argument("--seed", type=int, default=5143, help="A seed for reproducible training.")
parser.add_argument("--config", default='distillation.yaml', help="pruning config")
parser.add_argument("--temperature", default=1, type=float,
                    help='temperature parameter of distillation')
parser.add_argument("--loss_types", default=['CE', 'KL'], type=str, nargs='+',
                    help='loss types of distillation, should be a list of length 2, '
                    'first for student targets loss, second for teacher student loss.')
parser.add_argument("--loss_weights", default=[0.5, 0.5], type=float, nargs='+',
                    help='loss weights of distillation, should be a list of length 2, '
                    'and sum to 1.0, first for student targets loss weight, '
                    'second for teacher student loss weight.')
parser.set_defaults(augment=True)

def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    global args, best_prec1
    args, _ = parser.parse_known_args()
    best_prec1 = 0
    if args.seed is not None:
        set_seed(args.seed)
    if args.tensorboard: configure("runs/%s"%(args.name))

    # Data loading code
    normalize = transforms.Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2675, 0.2565, 0.2761])

    if args.augment:
        transform_train = transforms.Compose([
        	transforms.ToTensor(),
        	transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
        						(4,4,4,4),mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    # create teacher and student model
    if args.teacher_type == 'CNN-10':
        teacher_model = ConvNetMaker(plane_cifar100_book['10'])
    elif args.teacher_type == 'VGG-13':
        teacher_model = vgg.vgg13(num_classes=100)
    else:
        raise NotImplementedError('Unsupported teacher model type')
    teacher_model.load_state_dict(torch.load(args.teacher_model)['state_dict'])
    
    if args.student_type == 'CNN-2':
        student_model = ConvNetMaker(plane_cifar100_book['2'])
    elif args.student_type == 'VGG-8':
        student_model = vgg.VGG(vgg.make_layers([64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M']), num_classes=100)
    else:
        raise NotImplementedError('Unsupported student model type')

    # get the number of model parameters
    print('Number of teacher model parameters: {}'.format(
        sum([p.data.nelement() for p in teacher_model.parameters()])))
    print('Number of student model parameters: {}'.format(
        sum([p.data.nelement() for p in student_model.parameters()])))

    kwargs = {'num_workers': 0, 'pin_memory': True}
    assert(args.dataset == 'cifar100')
    train_dataset = datasets.__dict__[args.dataset.upper()]('../data', 
                                        train=True, download=True,
                                        transform=transform_train)
    # get logits of teacher model
    if args.loss_weights[1] > 0:
        from tqdm import tqdm
        def get_logits(teacher_model, train_dataset):
            print("***** Getting logits of teacher model *****")
            print(f"  Num examples = {len(train_dataset) }")
            logits_file = os.path.join(os.path.dirname(args.teacher_model), 'teacher_logits.npy')
            if not os.path.exists(logits_file):
                teacher_model.eval()
                train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, **kwargs)
                train_dataloader = tqdm(train_dataloader, desc="Evaluating")
                teacher_logits = []
                for step, (input, target) in enumerate(train_dataloader):
                    outputs = teacher_model(input)
                    teacher_logits += [x for x in outputs.numpy()]
                np.save(logits_file, np.array(teacher_logits))
            else:
                teacher_logits = np.load(logits_file)
            train_dataset.targets = [{'labels':l, 'teacher_logits':tl} \
                            for l, tl in zip(train_dataset.targets, teacher_logits)]
            return train_dataset
        with torch.no_grad():
            train_dataset = get_logits(teacher_model, train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.__dict__[args.dataset.upper()]('../data', train=False, transform=transform_test),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            student_model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # define optimizer
    optimizer = torch.optim.SGD(student_model.parameters(), args.lr,
                                momentum=args.momentum, nesterov = args.nesterov,
                                weight_decay=args.weight_decay)

    criterion = torch.nn.CrossEntropyLoss()

    # cosine learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*args.epochs)

    from neural_compressor.training import prepare_compression
    from neural_compressor.config import DistillationConfig, KnowledgeDistillationLossConfig
    distillation_criterion = KnowledgeDistillationLossConfig(temperature=args.temperature,
                                                             loss_types=args.loss_types,
                                                             loss_weights=args.loss_weights)
    conf = DistillationConfig(teacher_model, distillation_criterion)
    compression_manager = prepare_compression(student_model, conf)
    compression_manager.callbacks.on_train_begin()
    model = compression_manager.model
    train(train_loader, model, scheduler, compression_manager, best_prec1, criterion, val_loader)

    # change to framework model for further use
    model = model.model

def train(train_loader, model, scheduler, compression_manager, best_prec1, criterion, val_loader):
    for epoch in range(args.start_epoch, args.epochs):
        """Train for one epoch on the training set"""
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        # switch to train mode
        model.train()

        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            teacher_logits = None
            if isinstance(target, dict):
                teacher_logits = target['teacher_logits']
                target = target['labels']

            # compute output
            output = model(input)
            loss = criterion(output, target)
            loss = compression_manager.callbacks.on_after_compute_loss(input, output, loss, teacher_logits)

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target, topk=(1,))[0]
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # compute gradient and do SGD step
            compression_manager.callbacks.optimizer.zero_grad()
            loss.backward()
            compression_manager.callbacks.optimizer.step()
            scheduler.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'LR {scheduler._last_lr[0]:.6f}'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        loss=losses, top1=top1, scheduler=scheduler))

        compression_manager.callbacks.on_epoch_end()
        best_score = validate(val_loader, model, epoch + 1)
        # remember best prec@1 and save checkpoint
        is_best = best_score > best_prec1
        best_prec1 = max(best_score, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            }, is_best)
        # log to TensorBoard
        if args.tensorboard:
            log_value('train_loss', losses.avg, epoch)
            log_value('train_acc', top1.avg, epoch)
            log_value('learning_rate', scheduler._last_lr[0], epoch)


def validate(val_loader, model, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # compute output
        with torch.no_grad():
            output = model(input)

        # measure accuracy
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        log_value('val_acc', top1.avg, epoch)
    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pt'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.name) + 'model_best.pt')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
