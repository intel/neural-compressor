Step-by-Step
============

This document describes the step-by-step instructions for reproducing PyTorch ResNet50/ResNet18/ResNet101 tuning results with Intel® Neural Compressor.

# Prerequisite

### 1. Environment

PyTorch 1.8 or higher version is needed with pytorch_fx backend.

```Shell
cd examples/pytorch/image_recognition/torchvision_models/quantization/qat/fx
pip install -r requirements.txt
```
> Note: Validated PyTorch [Version](/docs/source/installation_guide.md#validated-software-environment).

### 2. Prepare Dataset

Download [ImageNet](http://www.image-net.org/) Raw image to dir: /path/to/imagenet.  The dir include below folder:

```bash
ls /path/to/imagenet
train  val
```

# Run

> Note: All torchvision model names can be passed as long as they are included in `torchvision.models`, below are some examples.

### 1. ResNet50

```Shell
python main.py -t -a resnet50 --pretrained --config /path/to/config_file /path/to/imagenet
```

### 2. ResNet18

```Shell
python main.py -t -a resnet18 --pretrained --config /path/to/config_file /path/to/imagenet
```

### 3. ResNext101_32x8d

```Shell
python main.py -t -a resnext101_32x8d --pretrained --config /path/to/config_file /path/to/imagenet
```
