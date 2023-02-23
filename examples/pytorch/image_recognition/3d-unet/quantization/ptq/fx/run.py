# coding=utf-8
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2020 - 2021 INTEL CORPORATION. All rights reserved.
# Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
sys.path.insert(0, os.getcwd())
import time

import argparse
import mlperf_loadgen as lg
import subprocess
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets


parser = argparse.ArgumentParser(description='PyTorch 3d-unet Training')
parser = argparse.ArgumentParser()
parser.add_argument("--backend",
                    choices=["pytorch", "onnxruntime", "tf", "ov"],
                    default="pytorch",
                    help="Backend")
parser.add_argument(
    "--scenario",
    choices=["SingleStream", "Offline", "Server", "MultiStream"],
    default="Offline",
    help="Scenario")
parser.add_argument("--accuracy",
                    action="store_true",
                    help="enable accuracy pass")
parser.add_argument("--mlperf_conf",
                    default="build/mlperf.conf",
                    help="mlperf rules config")
parser.add_argument("--user_conf",
                    default="user.conf",
                    help="user config for user LoadGen settings such as target QPS")
parser.add_argument(
    "--model_dir",
    default=
    "build/result/nnUNet/3d_fullres/Task043_BraTS2019/nnUNetTrainerV2__nnUNetPlansv2.mlperf.1",
    help="Path to the directory containing plans.pkl")
parser.add_argument("--model", help="Path to the ONNX, OpenVINO, or TF model")
parser.add_argument("--preprocessed_data_dir",
                    default="build/preprocessed_data",
                    help="path to preprocessed data")
parser.add_argument("--performance_count",
                    type=int,
                    default=16,
                    help="performance count")
parser.add_argument('--tune', dest='tune', action='store_true',
                    help='tune best int8 model on calibration dataset')
parser.add_argument('--performance', dest='performance', action='store_true',
                    help='run benchmark')
parser.add_argument('--int8', dest='int8', action='store_true',
                    help='run benchmark for int8')
args = parser.parse_args()


scenario_map = {
    "SingleStream": lg.TestScenario.SingleStream,
    "Offline": lg.TestScenario.Offline,
    "Server": lg.TestScenario.Server,
    "MultiStream": lg.TestScenario.MultiStream
}


def eval_func(model, args):
    
    if args.backend == "pytorch":
        from pytorch_SUT import get_pytorch_sut
        sut = get_pytorch_sut(model, args.preprocessed_data_dir,
                              args.performance_count)
    elif args.backend == "onnxruntime":
        from onnxruntime_SUT import get_onnxruntime_sut
        sut = get_onnxruntime_sut(args.model, args.preprocessed_data_dir,
                                  args.performance_count)
    elif args.backend == "tf":
        from tf_SUT import get_tf_sut
        sut = get_tf_sut(args.model, args.preprocessed_data_dir,
                         args.performance_count)
    elif args.backend == "ov":
        from ov_SUT import get_ov_sut
        sut = get_ov_sut(args.model, args.preprocessed_data_dir,
                         args.performance_count)
    else:
        raise ValueError("Unknown backend: {:}".format(args.backend))

    settings = lg.TestSettings()
    settings.scenario = scenario_map[args.scenario]
    settings.FromConfig(args.mlperf_conf, "3d-unet", args.scenario)
    settings.FromConfig(args.user_conf, "3d-unet", args.scenario)

    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
    else:
        settings.mode = lg.TestMode.PerformanceOnly

    log_path = "build/logs"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = log_path
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings

    print("Running Loadgen test...")
    if args.performance:
        start = time.time()
    lg.StartTestWithLogSettings(sut.sut, sut.qsl.qsl, settings, log_settings)
    if args.performance:
        end = time.time()

    if args.accuracy:
        print("Running accuracy script...")
        process = subprocess.Popen(['python3', 'accuracy-brats.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()

        print(out)
        print("Done!", float(err))

        if args.performance:
            print('Batch size = 1')
            print('Latency: %.3f ms' % ((end - start) * 1000 / sut.qsl.count))
            print('Throughput: %.3f images/sec' % (sut.qsl.count / (end - start)))
            print('Accuracy: {mean:.5f}'.format(mean=float(err)))

    print("Destroying SUT...")
    lg.DestroySUT(sut.sut)

    print("Destroying QSL...")
    lg.DestroyQSL(sut.qsl.qsl)
    return float(err)

sys.path.insert(0, os.path.join(os.getcwd(), "nnUnet"))
from nnunet.training.model_restore import load_model_and_checkpoint_files
from neural_compressor.experimental import Quantization, common
import pickle

# define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss()

# Data loading code
traindir = os.path.join(args.preprocessed_data_dir, 'train')
valdir = os.path.join(args.preprocessed_data_dir, 'val')
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

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True, sampler=None)

val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)

def main():
    args = parser.parse_args()

    class CalibrationDL():
        def __init__(self):
            path = os.path.abspath(os.path.expanduser('./brats_cal_images_list.txt'))
            with open(path, 'r') as f:
                self.preprocess_files = [line.rstrip() for line in f]

            self.loaded_files = {}
            self.batch_size = 1

        def __getitem__(self, sample_id):
            file_name = self.preprocess_files[sample_id]
            print("Loading file {:}".format(file_name))
            with open(os.path.join('build/calib_preprocess/', "{:}.pkl".format(file_name)), "rb") as f:
                self.loaded_files[sample_id] = pickle.load(f)[0]
            # note that calibration phase does not care label, here we return 0 for label free case.
            return self.loaded_files[sample_id], 0

        def __len__(self):
            self.count = len(self.preprocess_files)
            return self.count


    assert args.backend == "pytorch"
    model_path = os.path.join(args.model_dir, "plans.pkl")
    assert os.path.isfile(model_path), "Cannot find the model file {:}!".format(model_path)
    trainer, params = load_model_and_checkpoint_files(args.model_dir, folds=1, fp16=False, checkpoint_name='model_final_checkpoint')
    trainer.load_checkpoint_ram(params[0], False)
    model = trainer.network

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

            # measure elapsed time
            if i >= args.warmup_iter:
                batch_time.update(time.time() - start)

            if i % args.print_freq == 0:
                progress.print(i)

            if args.iter > 0 and i >= (args.warmup_iter + args.iter - 1):
                break

        print('Batch size = %d' % args.batch_size)
        print('Accuracy: {top1:.5f} Accuracy@5 {top5:.5f}'
              .format(top1=(top1.avg / 100), top5=(top5.avg / 100)))

    return top1.avg


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


if __name__ == "__main__":
    main()
