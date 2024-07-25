#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Intel Corporation
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

import argparse
import os
import sys
import torch
import numpy as np

from pathlib import Path
from tqdm import tqdm


from yolov5.models.common import DetectMultiBackend
from yolov5.utils.callbacks import Callbacks
from yolov5.utils.dataloaders import create_dataloader
from yolov5.utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    Profile,
    check_dataset,
    check_img_size,
    check_requirements,
    check_yaml,
    coco80_to_coco91_class,
    colorstr,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    xywh2xyxy,
)
from yolov5.utils.metrics import ap_per_class, box_iou
from yolov5.utils.plots import output_to_target, plot_images, plot_val_study
from yolov5.utils.torch_utils import select_device, smart_inference_mode

from neural_compressor.tensorflow.utils import BaseModel, CpuInfo


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_location', type=str, default='/datasets/mnist', help='dataset path')
parser.add_argument('--input_model', type=str, default='yolov5s.pb', help='input model path(s)')
parser.add_argument('--output_model', type=str, default='yolov5s_int8.pb', help='output model path(s)')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=224, help='inference size (pixels)')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
parser.add_argument('--verbose', nargs='?', const=True, default=False, help='verbose output')
parser.add_argument('--project', default='evaluate/val-cls', help='save to project/name')
parser.add_argument('--name', default='exp', help='save to project/name')
parser.add_argument('--tune', action="store_true", help='whether to apply quantization')
parser.add_argument('--benchmark', action="store_true", help='whether to run benchmark')
parser.add_argument('--mode', type=str, default='performance', help='run performance or accuracy benchmark')
parser.add_argument('--iteration', type=int, default=100, help='iteration for calibration or evaluation')
args = parser.parse_args()

def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix.

    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

@smart_inference_mode()
def evaluate(
    model, # model.pt path(s)
    source=args.dataset_location,
    imgsz=640,  # inference size (pixels)
    conf_thres=0.001,  # confidence threshold
    iou_thres=0.6,  # NMS IoU threshold
    max_det=300,  # maximum detections per image
    task="val",  # train, val, test, speed or study
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    workers=8,  # max dataloader workers (per RANK in DDP mode)
    single_cls=False,  # treat as single-class dataset
    verbose=False,  # verbose output
    project=args.project,  # save to project/name
    name="exp",  # save to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    save_dir=Path(""),
    callbacks=Callbacks(),
    compute_loss=None,
):
    if isinstance(model, BaseModel):
        model.save("./yolov5s_eval.pb")
        model = "./yolov5s_eval.pb"
    device = select_device(device)

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels").mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    model = DetectMultiBackend(model, device=device)
    stride, pt = model.stride, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    device = model.device
    batch_size = 1  # export.py models default to batch-size 1
    LOGGER.info(f"Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models")
    
    # Data
    #data = check_dataset(yaml_path)  # check

    # Configure
    model.eval()
    nc = 1 if single_cls else 80  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
    pad, rect = (0.5, pt)  # square inference for benchmarks

    dataloader = create_dataloader(
        source,
        imgsz,
        batch_size,
        stride,
        single_cls,
        pad=pad,
        rect=rect,
        workers=workers,
        prefix=colorstr(f"{task}: "),
    )[0]

    seen = 0
    names = model.names if hasattr(model, "names") else model.module.names  # get class names
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    s = ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "P", "R", "mAP50", "mAP50-95")
    p, r, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dt = Profile(), Profile(), Profile()  # profiling times
    loss = torch.zeros(3, device=device)
    stats, ap, ap_class = [], [], []
    callbacks.run("on_val_start")
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar
    iters = -1 if args.mode == "accuracy" else args.iteration
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        if batch_i == iters:
            break

        callbacks.run("on_val_batch_start")
        with dt[0]:
            im = im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            _, _, height, width = im.shape  # batch size, channels, height, width

        # Inference
        with dt[1]:
            preds, train_out = model(im) if compute_loss else (model(im, augment=False), None)

        # Batch size 1 inference drops the batch dim
        if isinstance(preds, list):
            preds = preds[0]

        if preds.dim() == 2:
            preds=preds.unsqueeze(0)

        # Loss
        if compute_loss:
            loss += compute_loss(train_out, targets)[1]  # box, obj, cls

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        lb = []  # for autolabelling
        with dt[2]:
            preds = non_max_suppression(
                preds, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls, max_det=max_det
            )

        if args.benchmark:
            # Metrics
            for si, pred in enumerate(preds):
                labels = targets[targets[:, 0] == si, 1:]
                nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
                path, shape = Path(paths[si]), shapes[si][0]
                correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
                seen += 1

                if npr == 0:
                    if nl:
                        stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    continue

                # Predictions
                if single_cls:
                    pred[:, 5] = 0
                predn = pred.clone()
                scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

                # Evaluate
                if nl:
                    tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                    scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                    labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                    correct = process_batch(predn, labelsn, iouv)
                stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

                callbacks.run("on_val_image_end", pred, predn, path, names, im[si])


            callbacks.run("on_val_batch_end", batch_i, im, targets, paths, shapes, preds)

    if args.tune:
        return 1
    
    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        _, _, p, r, _, ap, ap_class = ap_per_class(*stats, plot=False, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class
    if nt.sum() == 0:
        LOGGER.warning(f"WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels")

    pf = "%22s" + "%11i" * 2 + "%11.4g" * 4  # print format

    # Print results per class
    if (verbose or (nc < 50)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x.t / seen * 1e3 for x in dt)  # images per second
    latency = t[2]
    if args.benchmark and args.mode == "performance":
        print("Batch size = {}".format(args.batch_size))
        print("Latency: {:.3f} ms".format(latency))
        print("Throughput: {:.3f} images/sec".format(1000/latency))

    # Return results
    model.float()  # for training
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return map50


def main():
    if args.tune:
        from neural_compressor.tensorflow import StaticQuantConfig, quantize_model

        excluded_conv_names = [
            "functional_16_1/tf_conv_1/sequential_1/conv2d_1/convolution",
            "functional_16_1/tf_conv_1_2/sequential_1_1/conv2d_1_1/convolution",
            "functional_16_1/tfc3_1/tf_conv_2_1/conv2d_2_1/convolution",
            "functional_16_1/tfc3_1/sequential_2_1/tf_bottleneck_1/tf_conv_5_1/conv2d_5_1/convolution",
            "functional_16_1/tfc3_1/tf_conv_3_1/conv2d_3_1/convolution",
            "functional_16_1/tfc3_1/tf_conv_4_1/conv2d_4_1/convolution"
        ]
        quant_config = StaticQuantConfig(weight_granularity="per_channel")
        local_dtype = "bf16" if CpuInfo().bf16 or os.getenv("FORCE_BF16") == "1" else "fp32"
        local_config = StaticQuantConfig(weight_dtype=local_dtype, act_dtype=local_dtype)
        for conv_name in excluded_conv_names:
            quant_config.set_local(conv_name, local_config)
            
        q_model = quantize_model(args.input_model, quant_config, calib_func=evaluate)
        q_model.save(args.output_model)

    if args.benchmark:
        if args.mode == 'performance':
            evaluate(args.input_model)
        elif args.mode == 'accuracy':
            map50 = evaluate(args.input_model)
            print("Batch size = %d" % args.batch_size)
            LOGGER.info("Accuracy: %.4g" % map50)


if __name__ == "__main__":
    main()