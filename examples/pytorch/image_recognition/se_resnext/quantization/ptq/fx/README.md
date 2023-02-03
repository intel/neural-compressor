Step-by-Step
============

This document is used to list steps of reproducing PyTorch se_resnext tuning zoo result.

> **Note**
>
> * PyTorch quantization implementation in imperative path has limitation on automatically execution. It requires to manually add QuantStub and DequantStub for quantizable ops, it also requires to manually do fusion operation.
> * Intel® Neural Compressor supposes user have done these two steps before invoking Intel® Neural Compressor interface.For details, please refer to https://pytorch.org/docs/stable/quantization.html

# Prerequisite

### 1. Installation

#### Python First

Recommend python 3.6 or higher version.

#### Install dependency

```
pip install -r requirements.txt
```

#### Install SE_ResNext model

```Shell
cd examples/pytorch/image_recognition/se_resnext/quantization/ptq/fx
python setup.py install
```

> **Note**
>
> Please don't install public pretrainedmodels package.

### 2. Prepare Dataset

Download [ImageNet](http://www.image-net.org/) Raw image to dir: /path/to/imagenet. The dir include below folder:

```bash
ls /path/to/imagenet
train  val
```

# Run

### SE_ResNext50_32x4d

```Shell
python examples/imagenet_eval.py \
          --data /path/to/imagenet \
          -a se_resnext50_32x4d \
          -b 128 \
          -j 1 \
          -t
```

# Original SE_ResNext README

Please refer [SE_ResNext README](SE_ResNext_README.md)
