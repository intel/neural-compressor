Step-by-Step
============

This document describes the step-by-step instructions for reproducing PyTorch PeleeNet quantization and benchmarking results with Intel® Neural Compressor.

> **Note**
>
> * PyTorch quantization implementation in imperative path has limitations on automatically execution. It's required to manually add QuantStub and DequantStub for quantizable ops, also to manually do fusion operation.
> * Intel® Neural Compressor supposes user have done these two steps before invoking Intel® Neural Compressor interface.
>   For details, please refer to https://pytorch.org/docs/stable/quantization.html

# Prerequisite
## 1. Environment
Python 3.6 or higher version is recommended.
The dependent packages are all in requirements, please install as following.
```shell
cd examples/pytorch/image_recognition/peleenet/quantization/ptq/fx
pip install -r requirements.txt
```
### 2. Prepare Dataset
Download [ImageNet](http://www.image-net.org/) Raw image to dir: /path/to/imagenet. The dir includes below folder:
```bash
ls /path/to/imagenet
train  val
```
### 3. Prepare Pretrained Model
Download [weights](https://github.com/Robert-JunWang/PeleeNet/tree/master/weights) to examples/pytorch/image_recognition/peleenet/quantization/ptq/fx/weights.

# Run
## 1. Quantization
```shell
cd examples/pytorch/image_recognition/peleenet/quantization/ptq/fx
python main.py --tune --pretrained -j 1 /path/to/imagenet --weights weights/peleenet_acc7208.pth.tar
or
```shell
sh run_quant.sh --dataset_location=/path/to/imagenet --input_model=weights/peleenet_acc7208.pth.tar
```
## 2. Benchmark
```bash
# int8
sh run_benchmark.sh --dataset_location=/path/to/imagenet --mode=performance --int8=true
# fp32
sh run_benchmark.sh --dataset_location=/path/to/imagenet --mode=performance --input_model=weights/peleenet_acc7208.pth.tar
```
