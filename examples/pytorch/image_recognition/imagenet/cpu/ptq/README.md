Step-by-Step
============

This document describes the step-by-step instructions for reproducing PyTorch ResNet50/ResNet18/ResNet101 tuning results with Intel® Low Precision Optimization Tool.

> **Note**
>
> PyTorch quantization implementation in imperative path has limitation on automatically execution.
> It requires to manually add QuantStub and DequantStub for quantizable ops, it also requires to manually do fusion operation.
> Intel® Low Precision Optimization Tool has no capability to solve this framework limitation. Intel® Low Precision Optimization Tool supposes user have done these two steps before invoking Intel® Low Precision Optimization Tool interface.
> For details, please refer to https://pytorch.org/docs/stable/quantization.html

# Prerequisite

### 1. Installation

  ```Shell
  pip install -r requirements.txt
  ```

### 2. Prepare Dataset

  Download [ImageNet](http://www.image-net.org/) Raw image to dir: /path/to/imagenet.


# Run

### 1. ResNet50

  ```Shell
  cd examples/pytorch/image_recognition/imagenet/cpu/PTQ
  python main.py -t -a resnet50 --pretrained /path/to/imagenet
  ```

### 2. ResNet18

  ```Shell
  cd examples/pytorch/image_recognition/imagenet/cpu/PTQ
  python main.py -t -a resnet18 --pretrained /path/to/imagenet
  ```

### 3. ResNext101_32x8d

  ```Shell
  cd examples/pytorch/image_recognition/imagenet/cpu/PTQ
  python main.py -t -a resnext101_32x8d --pretrained /path/to/imagenet
  ```

### 4. InceptionV3

  ```Shell
  cd examples/pytorch/image_recognition/imagenet/cpu/PTQ
  python main.py -t -a inception_v3 --pretrained /path/to/imagenet
  ```

### 5. Mobilenet_v2

  ```Shell
  cd examples/pytorch/image_recognition/imagenet/cpu/PTQ
  python main.py -t -a mobilenet_v2 --pretrained /path/to/imagenet
  ```

### 6. ResNet50 dump tensors for debug

```Shell
  cd examples/pytorch/image_recognition/imagenet/cpu/PTQ
  python main_dump_tensors.py -t -a resnet50 --pretrained /path/to/imagenet
```

# Saving and loading model:
* Saving model:  
Intel® Low Precision Optimization Tool will automatically save tuning configure and weights of model which meet target goal when tuning process.
* loading model:  
```python
model                 # fp32 model
from ilit.utils.pytorch import load
quantized_model = load(
    os.path.join(Path, 'best_configure.yaml'),
    os.path.join(Path, 'best_model_weights.pt'), model)
```

Examples of enabling Intel® Low Precision Optimization Tool auto tuning on PyTorch ResNet
=======================================================

This is a tutorial of how to enable a PyTorch classification model with Intel® Low Precision Optimization Tool.

# User Code Analysis

Intel® Low Precision Optimization Tool supports three usages:

1. User only provide fp32 "model", and configure calibration dataset, evaluation dataset and metric in model-specific yaml config file.
2. User provide fp32 "model", calibration dataset "q_dataloader" and evaluation dataset "eval_dataloader", and configure metric in tuning.metric field of model-specific yaml config file.
3. User specifies fp32 "model", calibration dataset "q_dataloader" and a custom "eval_func" which encapsulates the evaluation dataset and metric by itself.

As ResNet18/50/101 series are typical classification models, use Top-K as metric which is built-in supported by Intel® Low Precision Optimization Tool. So here we integrate PyTorch ResNet with Intel® Low Precision Optimization Tool by the first use case for simplicity.

### Write Yaml Config File

In examples directory, there is a template.yaml. We could remove most of items and only keep mandotory item for tuning. 


```
model:
  name: imagenet_ptq
  framework: pytorch

quantization:
  calibration:
    sampling_size: 300
    dataloader:
      dataset:
        ImageFolder:
          root: /path/to/calibration/dataset
      transform:
        RandomResizedCrop:
          size: 224
        RandomHorizontalFlip:
        ToTensor:
        Normalize:
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]

evaluation:
  accuracy:
    metric:
      topk: 1
    dataloader:
      batch_size: 30
      dataset:
        ImageFolder:
          root: /path/to/evaluation/dataset
      transform:
        Resize:
          size: 256
        CenterCrop:
          size: 224
        ToTensor:
        Normalize:
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]
  performance:
    configs:
      cores_per_instance: 4
      num_of_instance: 7
    dataloader:
      batch_size: 1
      dataset:
        ImageFolder:
          root: /path/to/evaluation/dataset
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
    relative:  0.01
  exit_policy:
    timeout: 0
  random_seed: 9527

```

Here we choose topk built-in metric and set accuracy target as tolerating 0.01 relative accuracy loss of baseline. The default tuning strategy is basic strategy. The timeout 0 means unlimited time for a tuning config meet accuracy target.

### Prepare

PyTorch quantization requires two manual steps:

1. Add QuantStub and DeQuantStub for all quantizable ops.
2. Fuse possible patterns, such as Conv + Relu and Conv + BN + Relu.

Torchvision provide quantized_model, so we didn't do these steps above for all torchvision models. Please refer [torchvision](https://github.com/pytorch/vision/tree/master/torchvision/models/quantization)

The related code please refer to examples/pytorch/image_recognition/imagenet/cpu/PTQ/main.py.

### Code Update

After prepare step is done, we just need update main.py like below.

```
model.eval()
model.module.fuse_model()
from ilit import Quantization
quantizer = Quantization("./conf.yaml")
q_model = quantizer(model)
```

The quantizer() function will return a best quantized model during timeout constrain.

### Dump tensors for debug

Intel® Low Precision Optimization Tool can dump every layer output tensor which you specify in evaluation. You just need to add some setting to yaml configure file as below:

```
tensorboard: true
```
The default value of "tensorboard" is "off".

For example: 
```
sh run_tuning_dump_tensor.sh --topology=resnet18 --dataset_location=<Dataset>
```
A "./runs" folder will be generated, for example 

```
ls runs/eval/
tune_0_acc0.73  tune_1_acc0.71 tune_2_acc0.72
```
"tune_0_acc0.73" means FP32 baseline is accuracy 0.73, and the best tune result is tune_2 with accuracy 0.72. You may want to compare them in tensorboard. It will demonstrate the output tensor and weight of each op in "Histogram", you can also find the tune config of each tuning run in "Text":  
```
tensorboard --bind_all --logdir_spec baseline:./runs/eval/tune_0_acc0.73,tune_2:././runs/eval/tune_2_acc0.72
```

