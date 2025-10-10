#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#

### This file is originally from: [mlcommons repo](https://github.com/mlcommons/inference/tree/r0.5/others/cloud/single_stage_detector/pytorch/infer.py)
import os
from argparse import ArgumentParser
from utils import DefaultBoxes, Encoder, COCODetection
from base_model import Loss
from utils import SSDTransformer
from ssd_r34 import SSD_R34
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import time
import numpy as np
import torch.fx.experimental.optimization as optimization

try:
    import intel_extension_for_pytorch as ipex
    from intel_extension_for_pytorch.quantization import prepare, convert
    from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig
except:
    assert False, "please install intel-extension-for-pytorch, support version higher than 1.10"

#disable for fp32, bf16
use_ipex = False


def parse_args():
    parser = ArgumentParser(description="Train Single Shot MultiBox Detector"
                                        " on COCO")
    parser.add_argument('--data', '-d', type=str, default='../coco',
                        help='path to test and training data files')
    parser.add_argument('--no-cuda', action='store_true',
                        help='use available GPUs')
    parser.add_argument('--seed', '-s', type=int,
                        help='manually set random seed for torch')
    parser.add_argument('--device', '-did', type=int,
                        help='device id')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                        help='set batch size of valuation, default is 32')
    parser.add_argument('--iteration', '-iter', type=int, default=None,
                        help='set the iteration of inference, default is None')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--threshold', '-t', type=float, default=0.20,
                        help='stop training early at threshold')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='path to model checkpoint file, default is None')
    parser.add_argument('--image-size', default=[1200,1200], type=int, nargs='+',
                        help='input image sizes (e.g 1400 1400,1200 1200')
    parser.add_argument('--strides', default=[3,3,2,2,2,2], type=int, nargs='+',
                        help='stides for ssd model must include 6 numbers')
    parser.add_argument('--use-fp16', action='store_true')
    parser.add_argument('--ipex', action='store_true', default=False,
                        help='use intel pytorch extension')
    parser.add_argument('--int8', action='store_true', default=False,
                        help='enable ipex int8 path')
    parser.add_argument('--jit', action='store_true', default=False,
                        help='enable ipex jit path')
    parser.add_argument('--calibration', action='store_true', default=False,
                        help='doing int8 calibration step')
    parser.add_argument('--configure', default='configure.json', type=str, metavar='PATH',
                        help='path to int8 configures, default file name is configure.json')
    parser.add_argument("--dummy", action='store_true',
                        help="using  dummu data to test the performance of inference")
    parser.add_argument('-w', '--warmup-iterations', default=0, type=int, metavar='N',
                        help='number of warmup iterations to run')
    parser.add_argument('--autocast', action='store_true', default=False,
                        help='enable autocast')
    parser.add_argument('--profile', action='store_true', default=False,
                        help='enable profile')
    parser.add_argument('--accuracy-mode', action='store_true', default=False,
                        help='enable accuracy mode')
    parser.add_argument('--benchmark', action='store_true', default=False,
                        help='enable throughput mode')
    parser.add_argument('--latency-mode', action='store_true', default=False,
                        help='enable latency mode')
    parser.add_argument('--tune', dest='tune', action='store_true',
                        help='tune best int8 model with Neural Compressor on calibration dataset')
    parser.add_argument("--tuned_checkpoint", default='./saved_results', type=str, metavar='PATH',
                         help='path to checkpoint tuned by Neural Compressor (default: ./)')
    return parser.parse_args()


def show_memusage(device=0):
    import gpustat
    gpu_stats = gpustat.GPUStatCollection.new_query()
    item = gpu_stats.jsonify()["gpus"][device]
    print("{}/{}".format(item["memory.used"], item["memory.total"]))

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

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

def dboxes_R34_coco(figsize,strides):
    ssd_r34=SSD_R34(81,strides=strides)
    synt_img=torch.rand([1,3]+figsize)
    _,_,feat_size =ssd_r34(synt_img, extract_shapes = True)
    steps=[(int(figsize[0]/fs[0]),int(figsize[1]/fs[1])) for fs in feat_size]
    # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
    scales = [(int(s*figsize[0]/300),int(s*figsize[1]/300)) for s in [21, 45, 99, 153, 207, 261, 315]]
    aspect_ratios =  [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes

def eval_ssd_r34_mlperf_coco(args):
    from coco import COCO
    # Check that GPUs are actually available
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    dboxes = dboxes_R34_coco(args.image_size, args.strides)

    encoder = Encoder(dboxes)

    val_trans = SSDTransformer(dboxes, (args.image_size[0], args.image_size[1]), val=True)

    if not args.dummy:
        val_annotate = os.path.join(args.data, "annotations/instances_val2017.json")
        val_coco_root = os.path.join(args.data, "val2017")

        cocoGt = COCO(annotation_file=val_annotate)
        val_coco = COCODetection(val_coco_root, val_annotate, val_trans)
        inv_map = {v:k for k,v in val_coco.label_map.items()}

        if args.accuracy_mode:
            val_dataloader = DataLoader(val_coco,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        sampler=None,
                                        num_workers=args.workers)
        else:
            val_dataloader = DataLoader(val_coco,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        sampler=None,
                                        num_workers=args.workers,
                                        drop_last=True)
        labelnum = val_coco.labelnum
    else:
        cocoGt = None
        encoder = None
        inv_map = None
        val_dataloader = None
        labelnum = 81

    ssd_r34 = SSD_R34(labelnum, strides=args.strides)

    if args.checkpoint:
        print("loading model checkpoint", args.checkpoint)
        od = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
        ssd_r34.load_state_dict(od["model"])

    if use_cuda:
        ssd_r34.cuda(args.device)

    def coco_eval(model):
        from pycocotools.cocoeval import COCOeval
        device = args.device
        threshold = args.threshold
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        model.eval()

        ret = []

        inference_time = AverageMeter('InferenceTime', ':6.3f')
        decoding_time = AverageMeter('DecodingTime', ':6.3f')

        if (args.tune or args.accuracy_mode or args.benchmark or args.latency_mode) is False:
            print("one of --tune --accuracy-mode, --throughput-mode, --latency-mode must be input.")
            exit(-1)

        if args.accuracy_mode:
            if args.iteration is not None:
                print("accuracy mode should not input iteration")
            progress_meter_iteration = len(val_dataloader)
            epoch_number = 1
        else:
            if args.iteration is None:
                print("None accuracy mode must input --iteration")
                exit(-1)
            progress_meter_iteration = args.iteration
            epoch_number = (args.iteration // len(val_dataloader) + 1) if (args.iteration > len(val_dataloader)) else 1

        progress = ProgressMeter(
            progress_meter_iteration,
            [inference_time, decoding_time],
            prefix='Test: ')

        Profilling_iterator = 99
        start = time.time()

        if (args.accuracy_mode and args.int8) or (args.benchmark and arg.int8):
            for path, dirs, files in os.walk(args.tuned_checkpoint):
                if 'best_configure.json' in files:
                    args.int8_configure = os.path.join(path, 'best_configure.json')
                    break

        if args.tune:
            args.int8 = False if model.ipex_config_path is None else True

        if args.int8:
            print('runing int8 real inputs inference path')
            with torch.no_grad():
                total_iteration = 0
                for epoch in range(epoch_number):
                    for nbatch, (img, (img_id, img_size, bbox, label)) in enumerate(val_dataloader):
                        img = img.to(memory_format=torch.channels_last)
                        if total_iteration >= args.warmup_iterations:
                            start_time=time.time()

                        if args.profile and total_iteration == Profilling_iterator:
                            print("Profilling")
                            with torch.profiler.profile(on_trace_ready=torch.profiler.tensorboard_trace_handler("./int8_log")) as prof:
                                # ploc, plabel = model(img.to(memory_format=torch.channels_last))
                                ploc, plabel = model(img)
                            print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                        else:
                            #ploc, plabel = model(img.to(memory_format=torch.channels_last))
                            ploc, plabel = model(img)

                        if total_iteration >= args.warmup_iterations:
                            inference_time.update(time.time() - start_time)
                            end_time = time.time()
                        try:
                            results_raw = encoder.decode_batch(ploc, plabel, 0.50, 200,device=device)
                        except:
                            print("No object detected in nbatch: {}".format(total_iteration))
                            continue
                        if total_iteration >= args.warmup_iterations:
                            decoding_time.update(time.time() - end_time)

                        # Re-assembly the result
                        results = []
                        idx = 0
                        for i in range(results_raw[3].size(0)):
                            results.append((results_raw[0][idx:idx+results_raw[3][i]],
                                            results_raw[1][idx:idx+results_raw[3][i]],
                                            results_raw[2][idx:idx+results_raw[3][i]]))
                            idx += results_raw[3][i]

                        (htot, wtot) = [d.cpu().numpy() for d in img_size]
                        img_id = img_id.cpu().numpy()
                        # Iterate over batch elements
                        for img_id_, wtot_, htot_, result in zip(img_id, wtot, htot, results):
                            loc, label, prob = [r.cpu().numpy() for r in result]
                            # Iterate over image detections
                            for loc_, label_, prob_ in zip(loc, label, prob):
                                ret.append([img_id_, loc_[0]*wtot_, \
                                            loc_[1]*htot_,
                                            (loc_[2] - loc_[0])*wtot_,
                                            (loc_[3] - loc_[1])*htot_,
                                            prob_,
                                            inv_map[label_]])

                        if total_iteration % args.print_freq == 0:
                            progress.display(total_iteration)
                        if total_iteration == args.iteration:
                            break
                        total_iteration += 1
        else:
            if args.dummy:
                print('dummy inputs inference path is not supported')
            else:
                print('runing real inputs path')
                if args.autocast:
                    print('bf16 autocast enabled')
                    print('enable nhwc')
                    model = model.to(memory_format=torch.channels_last)
                    if use_ipex:
                        print('bf16 block format weights cache enabled')
                        model.model = ipex.optimize(model.model, dtype=torch.bfloat16, inplace=False)
                        # model = ipex.utils._convert_module_data_type(model, torch.bfloat16)
                    else:
                        from oob_utils import conv_bn_fuse
                        print('OOB bf16 conv_bn_fusion enabled')
                        model.model = conv_bn_fuse(model.model)

                    if args.jit:
                        if use_ipex:
                            print('enable IPEX jit path')
                            with torch.cpu.amp.autocast(), torch.no_grad():
                                model = torch.jit.trace(model, torch.randn(args.batch_size, 3, 1200, 1200).to(memory_format=torch.channels_last)).eval()
                        else:
                            print('enable OOB jit path')
                            with torch.cpu.amp.autocast(cache_enabled=False), torch.no_grad():
                                model = torch.jit.trace(model, torch.randn(args.batch_size, 3, 1200, 1200).to(memory_format=torch.channels_last)).eval()

                        model = torch.jit.freeze(model)
                        with torch.no_grad():
                            total_iteration = 0
                            for epoch in range(epoch_number):
                                for nbatch, (img, (img_id, img_size, bbox, label)) in enumerate(val_dataloader):
                                    with torch.no_grad():
                                        if total_iteration >= args.warmup_iterations:
                                            start_time=time.time()

                                        img = img.to(memory_format=torch.channels_last)
                                        if args.profile and total_iteration == Profilling_iterator:
                                            print("Profilling")
                                            with torch.profiler.profile(on_trace_ready=torch.profiler.tensorboard_trace_handler("./log")) as prof:
                                                ploc, plabel = model(img)
                                            print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                                        else:
                                            ploc, plabel = model(img)
                                        if total_iteration >= args.warmup_iterations:
                                            inference_time.update(time.time() - start_time)
                                            end_time = time.time()

                                        try:
                                            if args.profile and total_iteration == Profilling_iterator:
                                                with torch.profiler.profile(on_trace_ready=torch.profiler.tensorboard_trace_handler("./decode_log")) as prof:
                                                    results_raw = encoder.decode_batch(ploc, plabel, 0.50, 200, device=device)
                                                print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                                            else:
                                                results_raw = encoder.decode_batch(ploc, plabel, 0.50, 200, device=device)
                                        except:
                                            print("No object detected in nbatch: {}".format(total_iteration))
                                            continue
                                        if total_iteration >= args.warmup_iterations:
                                            decoding_time.update(time.time() - end_time)

                                        # Re-assembly the result
                                        results = []
                                        idx = 0
                                        for i in range(results_raw[3].size(0)):
                                            results.append((results_raw[0][idx:idx+results_raw[3][i]],
                                                            results_raw[1][idx:idx+results_raw[3][i]],
                                                            results_raw[2][idx:idx+results_raw[3][i]]))
                                            idx += results_raw[3][i]

                                        (htot, wtot) = [d.cpu().numpy() for d in img_size]
                                        img_id = img_id.cpu().numpy()

                                        for img_id_, wtot_, htot_, result in zip(img_id, wtot, htot, results):
                                            loc, label, prob = [r.cpu().numpy() for r in result]
                                            # Iterate over image detections
                                            for loc_, label_, prob_ in zip(loc, label, prob):
                                                ret.append([img_id_, loc_[0]*wtot_, \
                                                            loc_[1]*htot_,
                                                            (loc_[2] - loc_[0])*wtot_,
                                                            (loc_[3] - loc_[1])*htot_,
                                                            prob_,
                                                            inv_map[label_]])

                                        if total_iteration % args.print_freq == 0:
                                            progress.display(total_iteration)
                                        if total_iteration == args.iteration:
                                            break
                                        total_iteration += 1
                    else:
                        if use_ipex:
                            print('Ipex Autocast imperative path')
                            with torch.cpu.amp.autocast(), torch.no_grad():
                                total_iteration = 0
                                for epoch in range(epoch_number):
                                    for nbatch, (img, (img_id, img_size, bbox, label)) in enumerate(val_dataloader):
                                        with torch.no_grad():
                                            if total_iteration >= args.warmup_iterations:
                                                start_time=time.time()
                                            img = img.contiguous(memory_format=torch.channels_last)
                                            if args.profile and total_iteration == Profilling_iterator:
                                                print("Profilling")
                                                with torch.profiler.profile(on_trace_ready=torch.profiler.tensorboard_trace_handler("./bf16_imperative_log")) as prof:
                                                    ploc, plabel = model(img)
                                                print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                                            else:
                                                ploc, plabel = model(img)

                                            if total_iteration >= args.warmup_iterations:
                                                inference_time.update(time.time() - start_time)
                                                end_time = time.time()

                                            try:
                                                results_raw = encoder.decode_batch(ploc, plabel, 0.50, 200, device=device)
                                            except:
                                                print("No object detected in total_iteration: {}".format(total_iteration))
                                                continue
                                            if total_iteration >= args.warmup_iterations:
                                                decoding_time.update(time.time() - end_time)

                                            # Re-assembly the result
                                            results = []
                                            idx = 0
                                            for i in range(results_raw[3].size(0)):
                                                results.append((results_raw[0][idx:idx+results_raw[3][i]],
                                                                results_raw[1][idx:idx+results_raw[3][i]],
                                                                results_raw[2][idx:idx+results_raw[3][i]]))
                                                idx += results_raw[3][i]

                                            (htot, wtot) = [d.cpu().numpy() for d in img_size]
                                            img_id = img_id.cpu().numpy()

                                            for img_id_, wtot_, htot_, result in zip(img_id, wtot, htot, results):
                                                loc, label, prob = [r.cpu().numpy() for r in result]
                                                # Iterate over image detections
                                                for loc_, label_, prob_ in zip(loc, label, prob):
                                                    ret.append([img_id_, loc_[0]*wtot_, \
                                                                loc_[1]*htot_,
                                                                (loc_[2] - loc_[0])*wtot_,
                                                                (loc_[3] - loc_[1])*htot_,
                                                                prob_,
                                                                inv_map[label_]])

                                            if total_iteration % args.print_freq == 0:
                                                progress.display(total_iteration)
                                            if total_iteration == args.iteration:
                                                break
                                            total_iteration += 1
                        else:
                            print("OOB Autocast imperative path")
                            with torch.cpu.amp.autocast(), torch.no_grad():
                                total_iteration = 0
                                for epoch in range(epoch_number):
                                    for nbatch, (img, (img_id, img_size, bbox, label)) in enumerate(val_dataloader):
                                        if total_iteration >= args.warmup_iterations:
                                            start_time=time.time()
                                        img = img.contiguous(memory_format=torch.channels_last)
                                        if args.profile and total_iteration == Profilling_iterator:
                                            print("Profilling")
                                            with torch.profiler.profile(on_trace_ready=torch.profiler.tensorboard_trace_handler("./bf16_oob_log")) as prof:
                                                ploc, plabel = model(img)
                                            print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                                        else:
                                            ploc, plabel = model(img)

                                        if total_iteration >= args.warmup_iterations:
                                            inference_time.update(time.time() - start_time)
                                            end_time = time.time()

                                        with torch.cpu.amp.autocast(enabled=False):
                                            try:
                                                results_raw = encoder.decode_batch(ploc, plabel, 0.50, 200, device=device)
                                            except:
                                                print("No object detected in total_iteration: {}".format(total_iteration))
                                                continue
                                            if total_iteration >= args.warmup_iterations:
                                                decoding_time.update(time.time() - end_time)

                                            # Re-assembly the result
                                            results = []
                                            idx = 0
                                            for i in range(results_raw[3].size(0)):
                                                results.append((results_raw[0][idx:idx+results_raw[3][i]],
                                                                results_raw[1][idx:idx+results_raw[3][i]],
                                                                results_raw[2][idx:idx+results_raw[3][i]]))
                                                idx += results_raw[3][i]

                                            (htot, wtot) = [d.cpu().numpy() for d in img_size]
                                            img_id = img_id.cpu().numpy()

                                            for img_id_, wtot_, htot_, result in zip(img_id, wtot, htot, results):
                                                loc, label, prob = [r.cpu().numpy() for r in result]
                                                # Iterate over image detections
                                                for loc_, label_, prob_ in zip(loc, label, prob):
                                                    ret.append([img_id_, loc_[0]*wtot_, \
                                                                loc_[1]*htot_,
                                                                (loc_[2] - loc_[0])*wtot_,
                                                                (loc_[3] - loc_[1])*htot_,
                                                                prob_,
                                                                inv_map[label_]])

                                            if total_iteration % args.print_freq == 0:
                                                progress.display(total_iteration)
                                            if total_iteration == args.iteration:
                                                break
                                            total_iteration += 1
                else:
                    print('autocast disabled, fp32 is used')
                    print('enable nhwc')
                    model = model.to(memory_format=torch.channels_last)
                    if use_ipex:
                        print('fp32 block format weights cache enabled')
                        model.model = ipex.optimize(model.model, dtype=torch.float32, inplace=False)
                    if args.jit:
                        print("enable jit")
                        with torch.no_grad():
                            model = torch.jit.trace(model, torch.randn(args.batch_size, 3, 1200, 1200).to(memory_format=torch.channels_last))
                        model = torch.jit.freeze(model)
                    with torch.no_grad():
                        total_iteration = 0
                        for epoch in range(epoch_number):
                            for nbatch, (img, (img_id, img_size, bbox, label)) in enumerate(val_dataloader):
                                if total_iteration >= args.warmup_iterations:
                                    start_time=time.time()

                                img = img.contiguous(memory_format=torch.channels_last)
                                if args.profile and total_iteration == Profilling_iterator:
                                    print("Profilling")
                                    with torch.profiler.profile(on_trace_ready=torch.profiler.tensorboard_trace_handler("./fp32_log")) as prof:
                                        ploc, plabel = model(img)
                                    print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                                else:
                                    ploc, plabel = model(img)
                                if total_iteration >= args.warmup_iterations:
                                    inference_time.update(time.time() - start_time)
                                    end_time = time.time()
                                try:
                                    if args.profile and total_iteration == Profilling_iterator:
                                        with torch.profiler.profile(on_trace_ready=torch.profiler.tensorboard_trace_handler("./fp32_decode_log")) as prof:
                                            results_raw = encoder.decode_batch(ploc, plabel, 0.50, 200,device=device)
                                        print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                                    else:
                                        results_raw = encoder.decode_batch(ploc, plabel, 0.50, 200,device=device)
                                except:
                                    print("No object detected in total_iteration: {}".format(total_iteration))
                                    continue
                                if total_iteration >= args.warmup_iterations:
                                    decoding_time.update(time.time() - end_time)

                                # Re-assembly the result
                                results = []
                                idx = 0
                                for i in range(results_raw[3].size(0)):
                                    results.append((results_raw[0][idx:idx+results_raw[3][i]],
                                                    results_raw[1][idx:idx+results_raw[3][i]],
                                                    results_raw[2][idx:idx+results_raw[3][i]]))
                                    idx += results_raw[3][i]

                                (htot, wtot) = [d.cpu().numpy() for d in img_size]
                                img_id = img_id.cpu().numpy()
                                # Iterate over batch elements
                                for img_id_, wtot_, htot_, result in zip(img_id, wtot, htot, results):
                                    loc, label, prob = [r.cpu().numpy() for r in result]
                                    # Iterate over image detections
                                    for loc_, label_, prob_ in zip(loc, label, prob):
                                        ret.append([img_id_, loc_[0]*wtot_, \
                                                    loc_[1]*htot_,
                                                    (loc_[2] - loc_[0])*wtot_,
                                                    (loc_[3] - loc_[1])*htot_,
                                                    prob_,
                                                    inv_map[label_]])

                                if total_iteration % args.print_freq == 0:
                                    progress.display(total_iteration)
                                if total_iteration == args.iteration:
                                    break
                                total_iteration += 1
        print("Predicting Ended, total time: {:.2f} s".format(time.time()-start))

        batch_size = args.batch_size
        latency = inference_time.avg / batch_size * 1000
        perf = batch_size / inference_time.avg
        print('inference latency %.2f ms'%latency)
        print('inference performance %.2f fps'%perf)

        if not args.dummy:
            latency = decoding_time.avg / batch_size * 1000
            perf = batch_size / decoding_time.avg
            print('decoding latency %.2f ms'%latency)
            print('decodingperformance %.2f fps'%perf)

            total_time_avg = inference_time.avg + decoding_time.avg
            throughput = batch_size / total_time_avg
            print("Throughput: {:.3f} fps".format(throughput))

            cocoDt = cocoGt.loadRes(np.array(ret))

            E = COCOeval(cocoGt, cocoDt, iouType='bbox')
            E.evaluate()
            E.accumulate()
            E.summarize()
            print("Current AP: {:.5f} AP goal: {:.5f}".format(E.stats[0], threshold))
            print("Accuracy: {:.5f} ".format(E.stats[0]))

            return E.stats[0] #Average Precision  (AP) @[ IoU=050:0.95 | area=   all | maxDets=100 ]
        else:
            total_time_avg = inference_time.avg
            throughput = batch_size / total_time_avg
            print("Throughput: {:.3f} fps".format(throughput))
            return False

    if args.tune:
        ssd_r34.eval()
        ssd_r34.model = optimization.fuse(ssd_r34.model)

        from neural_compressor import quantization
        from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion
        tuning_criterion = TuningCriterion(max_trials=600)

        conf = PostTrainingQuantConfig(approach="static", backend="ipex",
            calibration_sampling_size=[800], tuning_criterion=tuning_criterion)
        q_model = quantization.fit(ssd_r34,
            conf=conf,
            eval_func=coco_eval,
            calib_dataloader=val_dataloader
        )
        q_model.save(args.tuned_checkpoint)
        return

    if args.benchmark or args.accuracy_mode:
        ssd_r34.eval()
        if args.int8:
            config_file = os.path.join(args.tuned_checkpoint, "best_model.pt")
            assert os.path.exists(config_file), "there is no ipex model file, Please tune with Neural Compressor first!"
            from neural_compressor.utils.pytorch import load
            ssd_r34 = load(args.tuned_checkpoint, ssd_r34)
        else:
            from neural_compressor.adaptor.pytorch import get_example_inputs
            example_inputs = get_example_inputs(ssd_r34, val_dataloader)
            ssd_r34 = ipex.optimize(ssd_r34)
            with torch.no_grad():
                ssd_r34 = torch.jit.trace(ssd_r34, example_inputs)
                ssd_r34 = torch.jit.freeze(ssd_r34)
        from neural_compressor.config import BenchmarkConfig
        from neural_compressor import benchmark
        b_conf = BenchmarkConfig(cores_per_instance=4, num_of_instance=1)
        benchmark.fit(ssd_r34, conf=b_conf, b_func=coco_eval)
        return




def main():
    args = parse_args()

    print(args)
    if not os.path.isdir('./models'):
        os.mkdir('./models')

    if args.seed is not None:
        print("Using seed = {}".format(args.seed))
        torch.manual_seed(args.seed)
        np.random.seed(seed=args.seed)
    if not args.no_cuda:
        torch.cuda.set_device(args.device)
        torch.backends.cudnn.benchmark = True
    else:
        if not "USE_IPEX" in os.environ:
            print('Set environment variable "USE_IPEX" to 1 (export USE_IPEX=1) to utilize Intel® Extension for PyTorch, if needed.')
    eval_ssd_r34_mlperf_coco(args)

if __name__ == "__main__":
    main()
