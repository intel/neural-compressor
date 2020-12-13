Step-by-Step
============

This document describes the step-by-step instructions for reproducing PyTorch PeleeNet tuning results with Intel® Low Precision Optimization Tool.

> **Note**
>
> PyTorch quantization implementation in imperative path has limitation on automatically execution.
> It requires to manually add QuantStub and DequantStub for quantizable ops, it also requires to manually do fusion operation.
> Intel® Low Precision Optimization Tool has no capability to solve this framework limitation. Intel® Low Precision Optimization Tool supposes user have done these two steps before invoking Intel® Low Precision Optimization Tool interface.
> For details, please refer to https://pytorch.org/docs/stable/quantization.html

# Prerequisite

### 1. Installation

  ```Shell
  # Install
  cd examples/pytorch/image_recognition/peleenet
  pip install -r requirements.txt
  ```

### 2. Prepare Dataset

  Download [ImageNet](http://www.image-net.org/) Raw image to dir: /path/to/imagenet.

### 3. Prepare pretrained model
  Download [weights](https://github.com/Robert-JunWang/PeleeNet/tree/master/weights) to examples/pytorch/image_recognition/peleenet/weights.

# Run

  ```Shell
  cd examples/pytorch/image_recognition/peleenet
  python main.py --tune --pretrained -j 1 /path/to/imagenet
  ```

Examples of enabling Intel® Low Precision Optimization Tool auto tuning on PyTorch ResNet
=======================================================

This is a tutorial of how to enable a PyTorch classification model with Intel® Low Precision Optimization Tool.

# User Code Analysis

Intel® Low Precision Optimization Tool supports three usages:

1. User only provide fp32 "model", and configure calibration dataset, evaluation dataset and metric in model-specific yaml config file.

2. User provide fp32 "model", calibration dataset "q_dataloader" and evaluation dataset "eval_dataloader", and configure metric in tuning.metric field of model-specific yaml config file.

3. User specifies fp32 "model", calibration dataset "q_dataloader" and a custom "eval_func" which encapsulates the evaluation dataset and metric by itself.

As PeleeNet are typical classification models, use Top-K as metric and imagenet dataset which are built-in supported by Intel® Low Precision Optimization Tool. So here we integrate PyTorch PeleeNet with Intel® Low Precision Optimization Tool by the first use case for simplicity.

### Write Yaml Config File

In examples directory, there is a template.yaml. We could remove most of items and only keep mandotory item for tuning. 


```
#conf.yaml
model:                                               # mandatory. lpot uses this model name and framework name to decide where to save tuning history and deploy yaml.
  name: peleenet
  framework: pytorch                                 # mandatory. supported values are tensorflow, pytorch, or mxnet; allow new framework backend extension.

quantization:                                        # optional. tuning constraints on model-wise for advance user to reduce tuning space.
  calibration:
    sampling_size: 256                               # optional. default value is the size of whole dataset. used to set how many portions of calibration dataset is used. exclusive with iterations field.
    dataloader:
      batch_size: 256
      dataset:
        ImageFolder:
          root: /path/to/calibration/dataset         # NOTE: modify to calibration dataset location if needed
      transform:
        RandomResizedCrop:
            size: 224
        RandomHorizontalFlip:
        ToTensor:
        Normalize:
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]

evaluation:                                          # optional. required if user doesn't provide eval_func in lpot.Quantization.
  accuracy:                                          # optional. required if user doesn't provide eval_func in lpot.Quantization.
    metric:
      topk: 1                                        # built-in metrics are topk, map, f1, allow user to register new metric.
    dataloader:
      batch_size: 256
      dataset:
        ImageFolder:
          root: /path/to/evaluation/dataset          # NOTE: modify to evaluation dataset location if needed
      transform:
        Resize:
          size: 256
        CenterCrop:
          size: 224
        ToTensor:
        Normalize:
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]
  performance:                                       # optional. used to benchmark performance of passing model.
    configs:
      cores_per_instance: 4
      num_of_instance: 7
    dataloader:
      batch_size: 1
      dataset:
        ImageFolder:
          root: /path/to/evaluation/dataset          # NOTE: modify to evaluation dataset location if needed
      transform:
        Resize:
          size: 256
        CenterCrop:
          size: 224
        ToTensor:
        Normalize:
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]

tuning:
  accuracy_criterion:
    relative:  0.01                                  # optional. default value is relative, other value is absolute. this example allows relative accuracy loss: 1%.
  exit_policy:
    timeout: 0                                       # optional. tuning timeout (seconds). default value is 0 which means early stop. combine with max_trials field to decide when to exit.
  random_seed: 9527                                  # optional. random seed for deterministic tuning.

```

Here we choose topk built-in metric and set calibration dataloader, evaluation dataloader and accuracy target as tolerating 0.01 relative accuracy loss of baseline. The default tuning strategy is basic strategy. The timeout 0 means unlimited time as well as a tuning config meet accuracy target.

### Prepare

PyTorch quantization requires two manual steps:

1. Add QuantStub and DeQuantStub for all quantizable ops.
2. Fuse possible patterns, such as Conv + Relu and Conv + BN + Relu.

It's intrinsic limitation of PyTorch quantizaiton imperative path. No way to develop a code to automatically do that.

The related code changes please refer to examples/pytorch/image_recognition/peleenet/peleenet.py.

### Code Update

After prepare step is done, we just need update main.py like below.

```
model.module.fuse()
from lpot import Quantization
quantizer = Quantization("./conf.yaml")
q_model = quantizer(model)
```

The quantizer() function will return a best quantized model during timeout constrain.
