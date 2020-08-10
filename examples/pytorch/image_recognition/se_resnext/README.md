Step-by-Step
============

This document is used to list steps of reproducing PyTorch se_resnext iLiT tuning zoo result.

> **Note**
>
> 1. PyTorch quantization implementation in imperative path has limitation on automatically execution.
> It requires to manually add QuantStub and DequantStub for quantizable ops, it also requires to manually do fusion operation.
> iLiT has no capability to solve this framework limitation. iLiT supposes user have done these two steps before invoking iLiT interface.
> For details, please refer to https://pytorch.org/docs/stable/quantization.html

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
  cd examples/pytorch/image_recognition/se_resnext
  python setup.py install
  ```
  > **Note**
  >
  > Please don't install public pretrainedmodels package.


### 2. Prepare Dataset

  Download [ImageNet](http://www.image-net.org/) Raw image to dir: /path/to/imagenet.

# Run

### SE_ResNext50_32x4d

  ```Shell
  cd examples/pytorch/image_recognition/se_resnext
  python examples/imagenet_eval.py \
            --data /path/to/imagenet \
            -a se_resnext50_32x4d \
            -b 128 \
            -j 1 \
            -t
  ```

Examples of enabling iLiT
=========================

This is a tutorial of how to enable SE_ResNext model with iLiT.

# User Code Analysis

iLiT supports three usages:

1. User only provide fp32 "model", and configure calibration dataset, evaluation dataset and metric in model-specific yaml config file.

2. User provide fp32 "model", calibration dataset "q_dataloader" and evaluation dataset "eval_dataloader", and configure metric in tuning.metric field of model-specific yaml config file.

3. User specifies fp32 "model", calibration dataset "q_dataloader" and a custom "eval_func" which encapsulates the evaluation dataset and metric by itself.


As SE_ResNext series are typical classification models, use Top-K as metric which is built-in supported by iLiT. So here we integrate PyTorch ResNet with iLiT by the first use case for simplicity.

### Write Yaml Config File
In examples directory, there is conf.yaml. We could remove most of items and only keep mandotory item for tuning.
```
framework:
  - name: pytorch

device: cpu

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
Here we set accuracy target as tolerating 0.01 relative accuracy loss of baseline. The default tuning strategy is basic strategy. The timeout 0 means unlimited time for a tuning config meet accuracy target.
> **Note** : iLiT tool don't support "mse" tuning strategy for pytorch framework

### Prepare
PyTorch quantization requires two manual steps:

  1. Add QuantStub and DeQuantStub for all quantizable ops.
  2. Fuse possible patterns, such as Conv + Relu and Conv + BN + Relu. In bert model, there is no fuse pattern.

It's intrinsic limitation of PyTorch quantizaiton imperative path. No way to develop a code to automatically do that.
The related code changes please refer to examples/pytorch/image_recognition/se_resnext/pretrainedmodels/models/senet.py.

### Code Update
After prepare step is done, we just need update imagenet_eval.py like below
```
if args.tune:
        model.eval()
        model.module.fuse_model()
        import ilit
        tuner = ilit.Tuner("./conf.yaml")
        q_model = tuner.tune(model)
        return
```
# Original SE_ResNext README
Please refer [SE_ResNext README](SE_ResNext_README.md)
