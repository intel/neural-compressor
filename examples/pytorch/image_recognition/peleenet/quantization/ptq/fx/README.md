Step-by-Step
============

This document describes the step-by-step instructions for reproducing PyTorch PeleeNet tuning results with Intel® Neural Compressor.

> **Note**
>
> * PyTorch quantization implementation in imperative path has limitation on automatically execution. It requires to manually add QuantStub and DequantStub for quantizable ops, it also requires to manually do fusion operation.
> * Intel® Neural Compressor supposes user have done these two steps before invoking Intel® Neural Compressor interface.
>   For details, please refer to https://pytorch.org/docs/stable/quantization.html

# Prerequisite

### 1. Installation

```Shell
# Install
cd examples/pytorch/image_recognition/peleenet/quantization/ptq/fx
pip install -r requirements.txt
```

### 2. Prepare Dataset

Download [ImageNet](http://www.image-net.org/) Raw image to dir: /path/to/imagenet. The dir include below folder:

```bash
ls /path/to/imagenet
train  val
```

### 3. Prepare pretrained model

Download [weights](https://github.com/Robert-JunWang/PeleeNet/tree/master/weights) to examples/pytorch/image_recognition/peleenet/quantization/ptq/fx/weights.

# Run

```Shell
cd examples/pytorch/image_recognition/peleenet/quantization/ptq/fx
python main.py --tune --pretrained -j 1 /path/to/imagenet
```