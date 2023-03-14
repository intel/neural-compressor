Step-by-Step
============

This document describes the step-by-step instructions for reproducing PyTorch se_resnext tuning and benchmarking results with Intel® Neural Compressor.

> **Note**
>
> * PyTorch quantization implementation in imperative path has limitation on automatically execution. It requires manually adding QuantStub and DequantStub for quantizable ops, and  also fusion operation manually.
> * Intel® Neural Compressor supposes user have done these two steps before invoking Intel® Neural Compressor interface.
>   For details, please refer to https://pytorch.org/docs/stable/quantization.html

# Prerequisite
## 1. Environment
Python 3.6 or higher version is recommended.
The dependent packages are all in requirements, please install as following.
```shell
cd examples/pytorch/image_recognition/se_resnext/quantization/ptq/fx
pip install -r requirements.txt
```
## 2. Install Model
```shell
python setup.py install
```
> **Note**
>
> Please don't install public pretrainedmodels package.
## 3. Prepare Dataset
Download [ImageNet](http://www.image-net.org/) Raw image to dir: /path/to/imagenet. The dir should include below folder:
```bash
ls /path/to/imagenet
train  val
```

# Run
## 1. Quantization
```shell
python examples/imagenet_eval.py \
          --data /path/to/imagenet \
          -a se_resnext50_32x4d \
          -b 128 \
          -j 1 \
          -t
```
## 2. Benchmark
```bash
# int8
sh run_benchmark.sh --int8=true --mode=performance --input_model=se_resnext50_32x4d  --dataset_location=/path/to/imagenet
# fp32
sh run_benchmark.sh --mode=performance --input_model=se_resnext50_32x4d  --dataset_location=/path/to/imagenet
```

# Original SE_ResNext README
Please refer [SE_ResNext README](SE_ResNext_README.md)
