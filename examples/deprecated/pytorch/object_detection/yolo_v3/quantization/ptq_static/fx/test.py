from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


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


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size, iter=0, warmup=0):
    batch_time = AverageMeter('Time', ':6.3f')
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        if batch_i >= warmup:
            end = time.time()
        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
        if batch_i >= warmup:
            batch_time.update(time.time() - end)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)
        if iter > 0 and batch_i >= iter + warmup - 1:
            break

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    print('Batch size = %d' % batch_size)
    if batch_size == 1:
        print('Latency: %.3f ms' % (batch_time.avg * 1000))
    print('Throughput: %.3f images/sec' % (batch_size / batch_time.avg))
    print('Accuracy: {mAP:.5f}'.format(mAP=AP.mean()))
    print(' Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)'
              .format(batch_time=batch_time))

    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou threshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument('-t', '--tune', dest='tune', action='store_true',
                        help='tune best int8 model on calibration dataset')
    parser.add_argument('-i', "--iter", default=0, type=int,
                        help='For accuracy measurement only.')
    parser.add_argument('-w', "--warmup_iter", default=5, type=int,
                        help='For benchmark measurement only.')
    parser.add_argument('--benchmark', dest='benchmark', action='store_true',
                        help='run benchmark')
    parser.add_argument('--accuracy_only', dest='accuracy_only', action='store_true',
                        help='run benchmark')
    parser.add_argument("--tuned_checkpoint", default='./saved_results', type=str, metavar='PATH',
                        help='path to checkpoint tuned by Neural Compressor (default: ./)')
    parser.add_argument('--int8', dest='int8', action='store_true',
                        help='run benchmark for int8')
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    if opt.tune:
        class yolo_dataLoader(object):
            def __init__(self, loader=None, model_type=None, device='cpu'):
                self.loader = loader
                self.device = device
                self.batch_size = loader.batch_size
            def __iter__(self):
                labels = []
                for _, imgs, targets in self.loader:
                    # Extract labels
                    labels += targets[:, 1].tolist()
                    # Rescale target
                    targets[:, 2:] = xywh2xyxy(targets[:, 2:])
                    targets[:, 2:] *= opt.img_size

                    Tensor = torch.FloatTensor
                    imgs = Variable(imgs.type(Tensor), requires_grad=False)
                    yield imgs, targets

        def eval_func(model):
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=opt.iou_thres,
                conf_thres=opt.conf_thres,
                nms_thres=opt.nms_thres,
                img_size=opt.img_size,
                batch_size=opt.batch_size,
            )
            return AP.mean()

        model.eval()
        dataset = ListDataset(valid_path, img_size=opt.img_size, augment=False, multiscale=False)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=opt.batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
        )
        nc_dataloader = yolo_dataLoader(dataloader)
        from neural_compressor import quantization
        from neural_compressor.config import PostTrainingQuantConfig
        conf = PostTrainingQuantConfig()
        q_model = quantization.fit(model,
                                   conf=conf,
                                   eval_func=eval_func,
                                   calib_dataloader=nc_dataloader
                                  )
        q_model.save(opt.tuned_checkpoint)
        exit(0)
  
    model.eval()
    if opt.int8:
        from neural_compressor.utils.pytorch import load
        new_model = load(
                os.path.abspath(os.path.expanduser(opt.tuned_checkpoint)), model)
    else:
        new_model = model
    if opt.benchmark:
        def b_func(model):
            return evaluate(
                model,
                path=valid_path,
                iou_thres=opt.iou_thres,
                conf_thres=opt.conf_thres,
                nms_thres=opt.nms_thres,
                img_size=opt.img_size,
                batch_size=opt.batch_size,
                iter=opt.iter,
                warmup=opt.warmup_iter
        )
        from neural_compressor.config import BenchmarkConfig
        from neural_compressor import benchmark
        b_conf = BenchmarkConfig(warmup=5,
                                 iteration=opt.iter,
                                 cores_per_instance=4,
                                 num_of_instance=1)
        benchmark.fit(new_model, b_conf, b_func=b_func)
        exit(0)
    elif opt.accuracy_only:
        evaluate(
                new_model,
                path=valid_path,
                iou_thres=opt.iou_thres,
                conf_thres=opt.conf_thres,
                nms_thres=opt.nms_thres,
                img_size=opt.img_size,
                batch_size=opt.batch_size,
                iter=opt.iter,
                warmup=opt.warmup_iter
                )
        exit(0)


    print("Compute mAP...")

    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=8,
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")
