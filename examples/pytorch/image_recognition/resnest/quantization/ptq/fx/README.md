Step-by-Step
============

This document describes the step-by-step instructions for reproducing PyTorch ResNest50 tuning results with Intel® Neural Compressor.

> **Note**
>
> * PyTorch quantization implementation in imperative path has limitation on automatically execution. It requires to manually add QuantStub and DequantStub for quantizable ops, it also requires to manually do fusion operation.
> * Intel® Neural Compressor supposes user have done these two steps before invoking Intel® Neural Compressor interface.
>   For details, please refer to https://pytorch.org/docs/stable/quantization.html

# Prerequisite

### 1. Installation

```Shell
cd examples/pytorch/image_recognition/resnest/quantization/ptq/fx
pip install -r requirements.txt
python setup.py install

```

### 2. Prepare Dataset

Download [ImageNet](http://www.image-net.org/) Raw image to dir: /path/to/imagenet. The dir include below folder:

```bash
ls /path/to/imagenet
train  val
```

# Run

### 1. ResNest50

```Shell
python -u scripts/torch/verify.py --tune --model resnest50 --batch-size what_you_want --workers 1 --no-cuda /path/to/imagenet
```