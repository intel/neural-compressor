from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel

import geffnet
from data import Dataset, create_loader, resolve_data_config
from utils import accuracy, AverageMeter

torch.backends.cudnn.benchmark = True

params_dict = {
    # Coefficients:   width,depth,res,dropout,crop-pct
    'efficientnet_b0': (1.0, 1.0, 224, 0.2, 0.875),
    'efficientnet_b1': (1.0, 1.1, 240, 0.2, 0.882),
    'efficientnet_b2': (1.1, 1.2, 260, 0.3, 0.890),
    'efficientnet_b3': (1.2, 1.4, 300, 0.3, 0.904),
    'tf_efficientnet_b4': (1.4, 1.8, 380, 0.4, 0.922),
    'tf_efficientnet_b5': (1.6, 2.2, 456, 0.4, 0.934),
    'tf_efficientnet_b6': (1.8, 2.6, 528, 0.5, 0.942),
    'tf_efficientnet_b7': (2.0, 3.1, 600, 0.5, 0.949),
    'tf_fficientnet_b8': (2.2, 3.6, 672, 0.5, 0.954),
    'mobilenetv3_rw': (None, None, 224, None, 0.875),
}

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--model', '-m', metavar='MODEL', default='spnasnet1_00',
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--crop-pct', type=float, default=None, metavar='PCT',
                    help='Override default crop pct of 0.875')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='Number classes in dataset')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--tf-preprocessing', dest='tf_preprocessing', action='store_true',
                    help='use tensorflow mnasnet preporcessing')
parser.add_argument('--no-cuda', dest='no_cuda', action='store_true',
                    help='')
parser.add_argument('--tune', action='store_true', help='int8 quantization tune with lpot')
parser.add_argument('-i', "--iter", default=0, type=int,
                    help='For accuracy measurement only.')
parser.add_argument('-w', "--warmup_iter", default=5, type=int,
                    help='For benchmark measurement only.')
parser.add_argument('--benchmark', dest='benchmark', action='store_true',
                    help='run benchmark')
parser.add_argument("--tuned_checkpoint", default='./saved_results', type=str, metavar='PATH',
                    help='path to checkpoint tuned by Low Precision Optimization Tool'
                         ' (default: ./)')
parser.add_argument('--int8', dest='int8', action='store_true',
                    help='run benchmark for int8')


def main():
    args = parser.parse_args()
    print(args)

    if args.img_size is None:
        args.img_size, args.crop_pct = get_image_size_crop_pct(args.model)

    if not args.checkpoint and not args.pretrained:
        args.pretrained = True

    if args.torchscript:
        geffnet.config.set_scriptable(True)

    # create model
    model = geffnet.create_model(
        args.model,
        num_classes=args.num_classes,
        in_chans=3,
        pretrained=args.pretrained,
        checkpoint_path=args.checkpoint)

    if args.torchscript:
        torch.jit.optimized_execution(True)
        model = torch.jit.script(model)

    print('Model %s created, param count: %d' %
          (args.model, sum([m.numel() for m in model.parameters()])))

    data_config = resolve_data_config(model, args)

    criterion = nn.CrossEntropyLoss()

    if not args.no_cuda:
        if args.num_gpu > 1:
            model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
        else:
            model = model.cuda()
        criterion = criterion.cuda()

    if args.tune:
        model.eval()
        model.fuse_model()
        conf_yaml = "conf_" + args.model + ".yaml"
        from lpot.experimental import Quantization, common
        quantizer = Quantization(conf_yaml)
        quantizer.model = common.Model(model)
        q_model = quantizer()
        q_model.save(args.tuned_checkpoint)
        exit(0)

    valdir = os.path.join(args.data, 'val')
    loader = create_loader(
        Dataset(valdir, load_bytes=args.tf_preprocessing),
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=not args.no_cuda,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        crop_pct=data_config['crop_pct'],
        tensorflow_preprocessing=args.tf_preprocessing)

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    model.fuse_model()
    if args.int8:
        from lpot.utils.pytorch import load
        new_model = load(
            os.path.abspath(os.path.expanduser(args.tuned_checkpoint)), model)
    else:
        new_model = model

    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            if i >= args.warmup_iterations:
                start = time.time()
            if not args.no_cuda:
                target = target.cuda()
                input = input.cuda()

            # compute output
            output = new_model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            if i >= args.warmup_iterations:
                # measure elapsed time
                batch_time.update(time.time() - start)

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}, {rate_avg:.3f}/s) \t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(loader), batch_time=batch_time,
                    rate_avg=input.size(0) / batch_time.avg,
                    loss=losses, top1=top1, top5=top5))
            if args.iterations > 0 and i >= args.iterations + args.warmup_iterations - 1:
                break

        print('Batch size = %d' % args.batch_size)
        if args.batch_size == 1:
            print('Latency: %.3f ms' % (batch_time.avg * 1000))
        print('Throughput: %.3f images/sec' % (args.batch_size / batch_time.avg))
        print('Accuracy: {top1:.5f} Accuracy@5 {top5:.5f}'
              .format(top1=(top1.avg / 100), top5=(top5.avg / 100)))


def get_image_size_crop_pct(model_name):
    if model_name in params_dict:
        _, _, res, _, crop_pct = params_dict[model_name]
    else:
        assert False, "Unsupported model:{}".format(model_name)
    return res, crop_pct


if __name__ == '__main__':
    main()
