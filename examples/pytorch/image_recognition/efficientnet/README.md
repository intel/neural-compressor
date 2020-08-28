Step-by-Step
============

This document describes the step-by-step instructions for reproducing PyTorch Efficientnet tuning results.

> **Note**
>
> PyTorch quantization implementation in imperative path has limitation on automatically execution.
> It requires to manually add QuantStub and DequantStub for quantizable ops, it also requires to manually do fusion operation.
> Intel® Low Precision Optimization Tool has no capability to solve this framework limitation. This tool supposes user have done these two steps before invoking its interface.
> For details, please refer to https://pytorch.org/docs/stable/quantization.html

# Prerequisite

### 1. Installation

  ```Shell
  cd examples/pytorch/image_recognition/efficientnet
  pip install -r requirements.txt
  python setup.py install
  ```

### 2. Prepare Dataset

  Download [ImageNet](http://www.image-net.org/) Raw image to dir: /path/to/imagenet.


# Run

### 1. Efficientnet_b0

  ```Shell
  cd examples/pytorch/image_recognition/efficientnet
  python validate.py --tune --model efficientnet_b0 --no-cuda --pretrained /path/to/imagenet
  ```

### 2. Mobilenetv3_rw

  ```Shell
  cd examples/pytorch/image_recognition/efficientnet
  python validate.py --tune --model mobilenetv3_rw --no-cuda --pretrained /path/to/imagenet
  ```

Examples of enabling Intel® Low Precision Optimization Tool auto tuning on PyTorch ResNet
=======================================================

This is a tutorial of how to enable a PyTorch classification model.

# User Code Analysis

Intel® Low Precision Optimization Tool supports three usages:

1. User only provide fp32 "model", and configure calibration dataset, evaluation dataset and metric in model-specific yaml config file.
2. User provide fp32 "model", calibration dataset "q_dataloader" and evaluation dataset "eval_dataloader", and configure metric in tuning.metric field of model-specific yaml config file.
3. User specifies fp32 "model", calibration dataset "q_dataloader" and a custom "eval_func" which encapsulates the evaluation dataset and metric by itself.

As Efficientnet series are typical classification models, use Top-K as metric which is built-in supported by Intel® Low Precision Optimization Tool. So here we integrate PyTorch Efficientnet with Intel® Low Precision Optimization Tool by the first use case for simplicity.

### Write Yaml Config File

In examples directory, there is a template.yaml. We could remove most of items and only keep mandotory item for tuning. 


```
framework:
  - name: pytorch                              # possible values are tensorflow, mxnet and pytorch

tuning:
  metric:
    topk: 1                                    # tuning metrics: accuracy 
  accuracy_criterion:
    - relative: 0.01                           # the tuning target of accuracy loss percentage: 1%
  timeout: 0                                   # tuning timeout (seconds)
  random_seed: 9527                            # random seed

calibration:
    dataloader:
      batch_size: 256
      dataset:
        - type: "ImageFolder"
        - root: "../imagenet/img/train" # NOTICE: config to your imagenet data path
      transform:
        RandomResizedCrop:
          - size: 224
        RandomHorizontalFlip:
        ToTensor:
        Normalize:
          - mean: [0.485, 0.456, 0.406]
          - std: [0.229, 0.224, 0.225]

evaluation:
  dataloader:
    batch_size: 256
    dataset:
      - type: "ImageFolder"
      - root: "../imagenet/img/val" # NOTICE: config to your imagenet data path
    transform:
      Resize:
        - size: 256
      CenterCrop:
        - size: 224
      ToTensor:
      Normalize:
        - mean: [0.485, 0.456, 0.406]
        - std: [0.229, 0.224, 0.225]
```

Here we choose topk built-in metric and set accuracy target as tolerating 0.01 relative accuracy loss of baseline. The default tuning strategy is basic strategy. The timeout 0 means unlimited time for a tuning config meet accuracy target.

### Prepare

PyTorch quantization requires two manual steps:

1. Add QuantStub and DeQuantStub for all quantizable ops.
2. Fuse possible patterns, such as Conv + Relu and Conv + BN + Relu.

The related code please refer to examples/pytorch/image_recognition/efficientnet/gen_efficientnet.py.py and efficientnet_builder.py.

### Code Update

After prepare step is done, we just need update main.py like below.

```
model.eval()
model.fuse_model()
import ilit
tuner = ilit.Tuner("./conf_efficientnet_b0.yaml")
q_model = tuner.tune(model)
```

The tune() function will return a best quantized model during timeout constrain.
