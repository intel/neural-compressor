Step-by-Step
============

This document describes the step-by-step instructions for reproducing PyTorch ResNet50/ResNet18/ResNet101 tuning results with iLiT.

> **Note**
>
> PyTorch quantization implementation in imperative path has limitation on automatically execution.
> It requires to manually add QuantStub and DequantStub for quantizable ops, it also requires to manually do fusion operation.
> iLiT has no capability to solve this framework limitation. iLiT supposes user have done these two steps before invoking iLiT interface.
> For details, please refer to https://pytorch.org/docs/stable/quantization.html

# Prerequisite

### 1. Installation

  ```Shell
  # Install iLiT
  pip install ilit

  # Install PyTorch 1.5.0
  pip install torch==1.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
  
  # Install Modified TorchVision
  cd examples/pytorch/vision
  python setup.py install
  ```

### 2. Prepare Dataset

  Download [ImageNet](http://www.image-net.org/) Raw image to dir: /path/to/imagenet.


# Run

### 1. ResNet50

  ```Shell
  cd examples/pytorch/image_recognition/resnet
  python main.py -t -a resnet50 --pretrained /path/to/imagenet
  ```

### 2. ResNet18

  ```Shell
  cd examples/pytorch/image_recognition/resnet
  python main.py -t -a resnet18 --pretrained /path/to/imagenet
  ```

### 3. ResNet101

  ```Shell
  cd examples/pytorch/image_recognition/resnet
  python main.py -t -a resnet101 --pretrained /path/to/imagenet
  ```

Examples of enabling iLiT auto tuning on PyTorch ResNet
=======================================================

This is a tutorial of how to enable a PyTorch classification model with iLiT.

# User Code Analysis

iLiT supports two usages:

1. User specifies fp32 "model", calibration dataset "q_dataloader", evaluation dataset "eval_dataloader" and metric in tuning.metric field of model-specific yaml config file.

2. User specifies fp32 "model", calibration dataset "q_dataloader" and a custom "eval_func" which encapsulates the evaluation dataset and metric by itself.

As ResNet18/50/101 series are typical classification models, use Top-K as metric which is built-in supported by iLiT. So here we integrate PyTorch ResNet with iLiT by the first use case for simplicity.

### Write Yaml config file

In examples directory, there is a template.yaml. We could remove most of items and only keep mandotory item for tuning. 


```
#conf.yaml

framework:
  - name: pytorch

tuning:
    metric:
      - topk: 1
    accuracy_criterion:
      - relative: 0.01
    timeout: 0
    random_seed: 9527
```

Here we choose topk built-in metric and set accuracy target as tolerating 0.01 relative accuracy loss of baseline. The default tuning strategy is basic strategy. The timeout 0 means early stop as well as a tuning config meet accuracy target.

### prepare

PyTorch quantization requires two manual steps:

1. Add QuantStub and DeQuantStub for all quantizable ops.
2. Fuse possible patterns, such as Conv + Relu and Conv + BN + Relu.

It's intrinsic limitation of PyTorch quantizaiton imperative path. No way to develop a code to automatically do that.

The related code changes please refer to examples/pytorch/vision/torchvision/models/resnet.py and fuse_resnext_modules() in main.py.

### code update

After prepare step is done, we just need update main.py like below.

```
import ilit
tuner = ilit.Tuner("./conf.yaml")
q_model = tuner.tune(model, train_loader, eval_dataloader=val_loader)
```

The iLiT tune() function will return a best quantized model during timeout constrain.
