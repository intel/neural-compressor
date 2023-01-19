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
from re import M

from utils import DefaultBoxes, Encoder, COCODetection
from base_model import Loss
from utils import SSDTransformer
from ssd_r34 import SSD_R34
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import time
import numpy as np
import torch.nn as nn
from torch.utils import ThroughputBenchmark
import threading
import torch.fx.experimental.optimization as optimization

use_ipex = False
if os.environ.get('USE_IPEX') == "1":
    import intel_extension_for_pytorch as ipex
    use_ipex = True

def get_bs_per_stream(batch_size, stream_number):
    # Follow the logic of multi stream module
    result = []
    batch_per_instance = batch_size // stream_number

    if batch_per_instance >= 1:
        # The input batchsize larger or equal to num_streams.
        used_num_streams = stream_number
        # If input batchsize larger than num_streams and not divisible,
        # the first remainder streams will have (mini_batch + 1) input size.
        instance_need_extra_input = batch_size % stream_number
    else:
        # The input batchsize less than num_streams,
        # only the first batchsize stream will have mini_batch(1) input.
        batch_per_instance = 1
        used_num_streams = batch_size
        instance_need_extra_input = 0
    start_idx = 0
    end_idx = 0
    for j in range(used_num_streams):
        if j < instance_need_extra_input:
            # Tail case, when the input image size larger than num_streams and not divisible,
            # the first remainder streams will have (mini_batch + 1) input size.
            end_idx = end_idx + (batch_per_instance + 1)
        else:
            # Input image size divisible of num_streams or input image size less than num_streams.
            end_idx = end_idx + batch_per_instance
        result.append(end_idx-start_idx)
        start_idx = end_idx
    return result

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
    parser.add_argument('--number-instance', default=7, type=int,
                        help='the instance number for throughput benchmark')
    parser.add_argument('--use-throughput-benchmark', action='store_true', default=False,
                        help='use throughput benchmark')
    parser.add_argument('--use-multi-stream-module', action='store_true', default=False,
                        help='use multi stream module')
    parser.add_argument('--instance-number', default=-1, type=int,
                    help='the instance number to run under runtime api')
    parser.add_argument('--async-execution', action='store_true', default=False,
                        help='use multi stream module')
    parser.add_argument('--start-core', default=-1, type=int,
                    help='the start core to creat cpu pool')
    parser.add_argument('--end-core', default=-1, type=int,
                    help='the end core to creat cpu pool')
    parser.add_argument('--accuracy-mode', action='store_true', default=False,
                        help='enable accuracy mode')
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

class SSD_R34_NMS(nn.Module):
    def __init__(self, model = None, encoder = None):

        super(SSD_R34_NMS, self).__init__()
        assert model is not None
        assert encoder is not None
        self.model = model
        self.encoder = encoder

    def forward(self, img):
        ploc, plabel = self.model(img)
        results = self.encoder.decode_batch(ploc, plabel, 0.5, 200, 0)
        return results

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

def coco_eval(model, val_dataloader, cocoGt, encoder, inv_map, args):
    from pycocotools.cocoeval import COCOeval
    device = args.device
    threshold = args.threshold
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    model.eval()

    if args.accuracy_mode:
        if args.iteration is not None:
            print("accuracy mode should not input iteration")
        epoch_number = 1
    else:
        if args.iteration is None:
            print("None accuracy mode must input --iteration")
            exit(-1)
        epoch_number = (args.iteration // len(val_dataloader) + 1) if (args.iteration > len(val_dataloader)) else 1

    ret = []

    inference_time = AverageMeter('InferenceTime', ':6.3f')
    decoding_time = AverageMeter('DecodingTime', ':6.3f')

    Profilling_iterator = 99
    start = time.time()
    if args.int8:
        model = model.eval()
        model_decode = SSD_R34_NMS(model, encoder)
        print('int8 conv_bn_fusion enabled')
        with torch.no_grad():
            model_decode.model.model = optimization.fuse(model_decode.model.model, inplace=False)

            if args.calibration:
                print("runing int8 LLGA calibration step not support in throughput benchmark")
            else:
                print("INT8 LLGA start trace")
                # insert quant/dequant based on configure.json
                conf = ipex.quantization.QuantConf(configure_file=args.configure)
                model_decode.eval()
                if args.use_multi_stream_module:
                    batch_per_stream = args.batch_size // args.number_instance
                    print("batch_per_stream for multi_stream_module is:", batch_per_stream)
                    model_decode = ipex.quantization.convert(model_decode, conf, torch.randn(batch_per_stream, 3, 1200, 1200).to(memory_format=torch.channels_last))
                else:
                    model_decode = ipex.quantization.convert(model_decode, conf, torch.randn(args.batch_size, 3, 1200, 1200).to(memory_format=torch.channels_last))
                print("done ipex default recipe.......................")
                # freeze the module
                # model = torch.jit._recursive.wrap_cpp_module(torch._C._freeze_module(model._c, preserveParameters=True))
                # model_decode = torch.jit._recursive.wrap_cpp_module(torch._C._freeze_module(model_decode._c, preserveParameters=True))

                # After freezing, run 1 time to warm up the profiling graph executor to insert prim::profile
                # At the 2nd run, the llga pass will be triggered and the model is turned into an int8 model: prim::profile will be removed and will have LlgaFusionGroup in the graph
                with torch.no_grad():
                    for i in range(2):
                        # _ = model_decode(torch.randn(args.batch_size, 3, 1200, 1200).to(memory_format=torch.channels_last))
                        if args.use_multi_stream_module:
                            _ = model_decode(torch.randn(batch_per_stream, 3, 1200, 1200).to(memory_format=torch.channels_last))
                        else:
                            _ = model_decode(torch.randn(args.batch_size, 3, 1200, 1200).to(memory_format=torch.channels_last))

                if args.use_throughput_benchmark:

                    print('runing int8 real inputs inference use_throughput_benchmark path')
                    bench = ThroughputBenchmark(model_decode)
                    if args.iteration > len(val_dataloader):
                        print("Using ThroughputBenchmark, but args.iteration:{0} larger than len(val_dataloader):{1}".format(args.iteration, len(val_dataloader)))
                        exit(-1)
                    for nbatch, (img, img_id, img_size, bbox, label) in enumerate(val_dataloader):
                        #bench.add_input(img.to(memory_format=torch.channels_last))
                        bench.add_input(img.to(memory_format=torch.channels_last))
                        if nbatch == args.iteration:
                            break

                    if args.profile:
                        print("Profilling")
                        with torch.profiler.profile(on_trace_ready=torch.profiler.tensorboard_trace_handler("./int8_log")) as prof:
                            # ploc, plabel = model(img)
                            print("start to running the benchmark")
                            print(args.number_instance)
                            stats = bench.benchmark(num_calling_threads=args.number_instance, num_warmup_iters=args.warmup_iterations, num_iters=args.iteration) #num_instance, warm up iters, total iters
                        # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                    else:
                        # ploc, plabel = model(img)
                        with torch.no_grad():
                            print("start to running the benchmark")
                            print(args.number_instance)
                            stats = bench.benchmark(num_calling_threads=args.number_instance, num_warmup_iters=args.warmup_iterations, num_iters=args.iteration) #num_instance, warm up iters, total iters
                elif args.use_multi_stream_module:
                    print('runing int8 real inputs inference use_multi_stream_module path. async_execution is:{}'.format(args.async_execution))
                    if args.async_execution:
                        if args.start_core != -1 and args.end_core != -1:
                            cpu_pool = ipex.cpu.runtime.CPUPool(core_ids=range(args.start_core, args.end_core + 1))
                        elif args.instance_number != -1:
                            cpu_pool = ipex.cpu.runtime.CPUPool(node_id=args.instance_number)
                        else:
                            print("args.instance_number or (args.start_core, args.end_core) must be indicated to create cpu_pool")
                            exit(-1)
                        model_decode = ipex.cpu.runtime.MultiStreamModule(model_decode, num_streams=args.number_instance, cpu_pool=cpu_pool, concat_output=False)
                    time_consume = 0
                    total_iteration = 0
                    for _ in range(epoch_number):
                        for nbatch, (img, img_id, img_size, bbox, label) in enumerate(val_dataloader):
                            img = img.to(memory_format=torch.channels_last)
                            if total_iteration >= args.warmup_iterations:
                                start_time=time.time()
                            result_raw = model_decode(img)
                            if total_iteration >= args.warmup_iterations:
                                step_time = time.time() - start_time
                                print("time of step {0} is: {1}".format(total_iteration, step_time))
                                time_consume += step_time

                            if args.accuracy_mode:
                                results = []
                                idx = 0
                                if args.async_execution:
                                    mini_batch_of_each_stream = get_bs_per_stream(img.size(0), args.number_instance)
                                    used_num_streams = mini_batch_of_each_stream.__len__()
                                    for i in range(used_num_streams):
                                        idx2 = 0
                                        mini_bs_of_this_stream = mini_batch_of_each_stream[i]
                                        for j in range(mini_bs_of_this_stream):
                                            results.append((result_raw[i][0][idx2:idx2+result_raw[i][3][j]],
                                                            result_raw[i][1][idx2:idx2+result_raw[i][3][j]],
                                                            result_raw[i][2][idx2:idx2+result_raw[i][3][j]]))
                                            idx2 += result_raw[i][3][j]
                                else:
                                    for i in range(result_raw[3].size(0)):
                                        results.append((result_raw[0][idx:idx+result_raw[3][i]],
                                                        result_raw[1][idx:idx+result_raw[3][i]],
                                                        result_raw[2][idx:idx+result_raw[3][i]]))
                                        idx += result_raw[3][i]
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
                            if total_iteration == args.iteration:
                                fps = args.batch_size * (args.iteration - args.warmup_iterations + 1) / time_consume
                                avg_time = time_consume * 1000 / (args.iteration - args.warmup_iterations + 1)
                                print('Instance num: %d Avg Time/Iteration: %f msec Throughput: %f fps' %(args.instance_number, avg_time, fps))
                                break
                            total_iteration += 1
                else:
                    print('runing int8 real inputs inference pthread weight sharing path')
                    def run_model(m, tid):
                        time_consume = 0
                        with torch.no_grad():
                            total_iteration = 0
                            for _ in range(epoch_number):
                                for nbatch, (img, img_id, img_size, bbox, label) in enumerate(val_dataloader):
                                    img = img.to(memory_format=torch.channels_last)
                                    if total_iteration >= args.warmup_iterations:
                                        start_time=time.time()
                                    #m(img.to(memory_format=torch.channels_last))
                                    m(img)
                                    if total_iteration >= args.warmup_iterations:
                                        time_consume += time.time() - start_time
                                    if total_iteration == args.iteration:
                                        fps = (args.iteration - args.warmup_iterations + 1) / time_consume
                                        avg_time = time_consume * 1000 / (args.iteration - args.warmup_iterations + 1)
                                        print('Instance num: %d Avg Time/Iteration: %f msec Throughput: %f fps' %(tid, avg_time, fps))
                                        break
                                    total_iteration += 1
                    threads = []
                    for i in range(1, args.number_instance+1):
                        thread = threading.Thread(target=run_model, args=(model_decode, i))
                        threads.append(thread)
                        thread.start()
                    for thread in threads:
                        thread.join()
                    exit()

    else:
        if args.dummy:
            print('dummy inputs inference path is not supported')
        else:
            print('runing real inputs path')
            model_decode = SSD_R34_NMS(model, encoder)
            if args.autocast:
                print('bf16 autocast enabled')
                print('enable nhwc')
                # model = model.to(memory_format=torch.channels_last)
                model_decode.model = model_decode.model.to(memory_format=torch.channels_last)
                if use_ipex:
                    print('bf16 block format weights cache enabled')
                    # model.model = ipex.optimize(model.model, dtype=torch.bfloat16, level='O0')
                    model_decode.model.model = ipex.optimize(model_decode.model.model, dtype=torch.bfloat16, inplace=False)

                if args.jit:
                    print('enable jit')
                    with torch.cpu.amp.autocast(), torch.no_grad():
                        # model = torch.jit.trace(model, torch.randn(args.batch_size, 3, 1200, 1200).to(memory_format=torch.channels_last)).eval()
                        batch_per_stream = (args.batch_size // args.number_instance) if args.use_multi_stream_module else args.batch_size
                        model_decode = torch.jit.trace(model_decode, torch.randn(batch_per_stream, 3, 1200, 1200).to(memory_format=torch.channels_last)).eval()
                    # model = torch.jit.freeze(model)
                    model_decode = torch.jit.freeze(model_decode)

                    if args.use_throughput_benchmark:
                        print('bf16 throughput benchmark')
                        bench = ThroughputBenchmark(model_decode)
                        if args.iteration > len(val_dataloader):
                            print("Using ThroughputBenchmark, but args.iteration:{0} larger than len(val_dataloader):{1}".format(args.iteration, len(val_dataloader)))
                            exit(-1)
                        for nbatch, (img, img_id, img_size, bbox, label) in enumerate(val_dataloader):
                            bench.add_input(img.to(memory_format=torch.channels_last))
                            if nbatch == args.iteration:
                                break

                        with torch.no_grad():
                            print("start to running the benchmark")
                            print(args.number_instance)
                            stats = bench.benchmark(num_calling_threads=args.number_instance, num_warmup_iters=args.warmup_iterations, num_iters=args.iteration) #num_instance, warm up iters, total iters
                    elif args.use_multi_stream_module:
                        print('bf16 use_multi_stream_module path')
                        if args.async_execution:
                            if args.start_core != -1 and args.end_core != -1:
                                cpu_pool = ipex.cpu.runtime.CPUPool(core_ids=range(args.start_core, args.end_core + 1))
                            elif args.instance_number != -1:
                                cpu_pool = ipex.cpu.runtime.CPUPool(node_id=args.instance_number)
                            else:
                                print("args.instance_number or (args.start_core, args.end_core) must be indicated to create cpu_pool")
                                exit(-1)
                            model_decode = ipex.cpu.runtime.MultiStreamModule(model_decode, num_streams=args.number_instance, cpu_pool=cpu_pool, concat_output=False)
                        time_consume = 0
                        total_iteration = 0
                        for _ in range(epoch_number):
                            for nbatch, (img, img_id, img_size, bbox, label) in enumerate(val_dataloader):
                                img = img.to(memory_format=torch.channels_last)
                                if total_iteration >= args.warmup_iterations:
                                    start_time=time.time()
                                result_raw = model_decode(img)
                                if total_iteration >= args.warmup_iterations:
                                    step_time = time.time() - start_time
                                    print("time of step {0} is: {1}".format(total_iteration, step_time))
                                    time_consume += step_time

                                if args.accuracy_mode:
                                    results = []
                                    idx = 0
                                    if args.async_execution:
                                        mini_batch_of_each_stream = get_bs_per_stream(img.size(0), args.number_instance)
                                        used_num_streams = mini_batch_of_each_stream.__len__()
                                        for i in range(used_num_streams):
                                            idx2 = 0
                                            mini_bs_of_this_stream = mini_batch_of_each_stream[i]
                                            for j in range(mini_bs_of_this_stream):
                                                results.append((result_raw[i][0][idx2:idx2+result_raw[i][3][j]],
                                                                result_raw[i][1][idx2:idx2+result_raw[i][3][j]],
                                                                result_raw[i][2][idx2:idx2+result_raw[i][3][j]]))
                                                idx2 += result_raw[i][3][j]
                                    else:
                                        for i in range(result_raw[3].size(0)):
                                            results.append((result_raw[0][idx:idx+result_raw[3][i]],
                                                            result_raw[1][idx:idx+result_raw[3][i]],
                                                            result_raw[2][idx:idx+result_raw[3][i]]))
                                            idx += result_raw[3][i]
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
                                if total_iteration == args.iteration:
                                    fps = args.batch_size * (args.iteration - args.warmup_iterations + 1) / time_consume
                                    avg_time = time_consume * 1000 / (args.iteration - args.warmup_iterations + 1)
                                    print('Instance num: %d Avg Time/Iteration: %f msec Throughput: %f fps' %(args.instance_number, avg_time, fps))
                                    break
                                total_iteration += 1
                    else:
                        print('bf16 pthread weight sharing path')
                        def run_model(m, tid):
                            time_consume = 0
                            with torch.no_grad():
                                total_iteration = 0
                                for _ in range(epoch_number):
                                    for nbatch, (img, img_id, img_size, bbox, label) in enumerate(val_dataloader):
                                        if total_iteration >= args.warmup_iterations:
                                            start_time=time.time()
                                        img = img.to(memory_format=torch.channels_last)
                                        m(img)
                                        if total_iteration >= args.warmup_iterations:
                                            time_consume += time.time() - start_time
                                        if total_iteration == args.iteration:
                                            fps = (args.iteration - args.warmup_iterations + 1) / time_consume
                                            avg_time = time_consume * 1000 / (args.iteration - args.warmup_iterations + 1)
                                            print('Instance num: %d Avg Time/Iteration: %f msec Throughput: %f fps' %(tid, avg_time, fps))
                                            break
                                        total_iteration += 1
                        threads = []
                        for i in range(1, args.number_instance+1):
                            thread = threading.Thread(target=run_model, args=(model_decode, i))
                            threads.append(thread)
                            thread.start()
                        for thread in threads:
                            thread.join()
                        exit()

                else:
                    if use_ipex:
                        print('Ipex Autocast imperative path in throughput benchmark not support')
                    else:
                        print("OOB Autocast imperative path in throughput benchmark not support")
                        exit(-1)
            else:
                print('autocast disabled, fp32 is used')
                print('enable nhwc')
                model_decode.model = model_decode.model.to(memory_format=torch.channels_last)
                if use_ipex:
                    print('fp32 block format weights cache enabled')
                    model_decode.model.model = ipex.optimize(model_decode.model.model, dtype=torch.float32, inplace=False)
                if args.jit:
                    print("enable jit")
                    with torch.no_grad():
                        batch_per_stream = (args.batch_size // args.number_instance) if args.use_multi_stream_module else args.batch_size
                        model_decode = torch.jit.trace(model_decode, torch.randn(batch_per_stream, 3, 1200, 1200).to(memory_format=torch.channels_last)).eval()
                    model_decode = torch.jit.freeze(model_decode)
                    if args.use_throughput_benchmark:
                        print('fp32 throughput benchmark')
                        bench = ThroughputBenchmark(model_decode)
                        if args.iteration > len(val_dataloader):
                            print("Using ThroughputBenchmark, but args.iteration:{0} larger than len(val_dataloader):{1}".format(args.iteration, len(val_dataloader)))
                            exit(-1)
                        for nbatch, (img, img_id, img_size, bbox, label) in enumerate(val_dataloader):
                            bench.add_input(img.to(memory_format=torch.channels_last))
                            if nbatch == args.iteration:
                                break

                        with torch.no_grad():
                            print("start to running the benchmark")
                            print(args.number_instance)
                            stats = bench.benchmark(num_calling_threads=args.number_instance, num_warmup_iters=args.warmup_iterations, num_iters=args.iteration) #num_instance, warm up iters, total iters
                    elif args.use_multi_stream_module:
                        print('fp32 use_multi_stream_module path')
                        if args.async_execution:
                            if args.start_core != -1 and args.end_core != -1:
                                cpu_pool = ipex.cpu.runtime.CPUPool(core_ids=range(args.start_core, args.end_core + 1))
                            elif args.instance_number != -1:
                                cpu_pool = ipex.cpu.runtime.CPUPool(node_id=args.instance_number)
                            else:
                                print("args.instance_number or (args.start_core, args.end_core) must be indicated to create cpu_pool")
                                exit(-1)
                            model_decode = ipex.cpu.runtime.MultiStreamModule(model_decode, num_streams=args.number_instance, cpu_pool=cpu_pool, concat_output=False)
                        time_consume = 0
                        total_iteration = 0
                        for _ in range(epoch_number):
                            for nbatch, (img, img_id, img_size, bbox, label) in enumerate(val_dataloader):
                                img = img.to(memory_format=torch.channels_last)
                                if total_iteration >= args.warmup_iterations:
                                    start_time=time.time()
                                result_raw = model_decode(img)
                                if total_iteration >= args.warmup_iterations:
                                    step_time = time.time() - start_time
                                    print("time of step {0} is: {1}".format(total_iteration, step_time))
                                    time_consume += step_time

                                if args.accuracy_mode:
                                    results = []
                                    idx = 0
                                    if args.async_execution:
                                        mini_batch_of_each_stream = get_bs_per_stream(img.size(0), args.number_instance)
                                        used_num_streams = mini_batch_of_each_stream.__len__()
                                        for i in range(used_num_streams):
                                            idx2 = 0
                                            mini_bs_of_this_stream = mini_batch_of_each_stream[i]
                                            for j in range(mini_bs_of_this_stream):
                                                results.append((result_raw[i][0][idx2:idx2+result_raw[i][3][j]],
                                                                result_raw[i][1][idx2:idx2+result_raw[i][3][j]],
                                                                result_raw[i][2][idx2:idx2+result_raw[i][3][j]]))
                                                idx2 += result_raw[i][3][j]
                                    else:
                                        for i in range(result_raw[3].size(0)):
                                            results.append((result_raw[0][idx:idx+result_raw[3][i]],
                                                            result_raw[1][idx:idx+result_raw[3][i]],
                                                            result_raw[2][idx:idx+result_raw[3][i]]))
                                            idx += result_raw[3][i]
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
                                if total_iteration == args.iteration:
                                    fps = args.batch_size * (args.iteration - args.warmup_iterations + 1) / time_consume
                                    avg_time = time_consume * 1000 / (args.iteration - args.warmup_iterations + 1)
                                    print('Instance num: %d Avg Time/Iteration: %f msec Throughput: %f fps' %(args.instance_number, avg_time, fps))
                                    break
                                total_iteration += 1
                    else:
                        print('fp32 pthread weight sharing path')
                        def run_model(m, tid):
                            time_consume = 0
                            total_iteration = 0
                            for _ in range(epoch_number):
                                for nbatch, (img, img_id, img_size, bbox, label) in enumerate(val_dataloader):
                                    if total_iteration >= args.warmup_iterations:
                                        start_time=time.time()
                                    img = img.to(memory_format=torch.channels_last)
                                    m(img)
                                    if total_iteration >= args.warmup_iterations:
                                        time_consume += time.time() - start_time
                                    if total_iteration == args.iteration:
                                        fps = (args.iteration - args.warmup_iterations + 1) / time_consume
                                        avg_time = time_consume * 1000 / (args.iteration - args.warmup_iterations + 1)
                                        print('Instance num: %d Avg Time/Iteration: %f msec Throughput: %f fps' %(tid, avg_time, fps))
                                        break
                                    total_iteration += 1
                        threads = []
                        for i in range(1, args.number_instance+1):
                            thread = threading.Thread(target=run_model, args=(model_decode, i))
                            threads.append(thread)
                            thread.start()
                        for thread in threads:
                            thread.join()
                        exit()
                else:
                    print("FP32 Imperative path with weight sharing is not enabled")
                    exit(-1)
    if args.use_throughput_benchmark:
        print("Predicting Ended, total time: {:.2f} s".format(time.time()-start))
        batch_size = args.batch_size
        latency = stats.latency_avg_ms
        perf = stats.iters_per_second * batch_size
        print('inference latency %.2f ms'%latency)
        print('inference performance %.2f fps'%perf)

        if not args.dummy:
            print("Throughput: {:.3f} fps".format(perf))
        else:
            total_time_avg = inference_time.avg
            throughput = batch_size / total_time_avg
            print("Throughput: {:.3f} fps".format(throughput))
            return False
    elif args.use_multi_stream_module and not args.dummy and args.accuracy_mode:
        cocoDt = cocoGt.loadRes(np.array(ret))

        E = COCOeval(cocoGt, cocoDt, iouType='bbox')
        E.evaluate()
        E.accumulate()
        E.summarize()
        print("Current AP: {:.5f} AP goal: {:.5f}".format(E.stats[0], threshold))
        print("Accuracy: {:.5f} ".format(E.stats[0]))

        return (E.stats[0] >= threshold) #Average Precision  (AP) @[ IoU=050:0.95 | area=   all | maxDets=100 ]


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
    coco_eval(ssd_r34, val_dataloader, cocoGt, encoder, inv_map, args)

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

