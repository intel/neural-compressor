Step-by-Step
============

This document describes the step-by-step instructions for reproducing PyTorch ResNet50/ResNet18/ResNet101 tuning results with Intel® Low Precision Optimization Tool.

> **Note**
>
> * PyTorch quantization implementation in imperative path has limitation on automatically execution. It requires to manually add QuantStub and DequantStub for quantizable ops, it also requires to manually do fusion operation.
> * LPOT requires users to complete these two manual steps before triggering auto-tuning process.
>   For details, please refer to https://pytorch.org/docs/stable/quantization.html

# Prerequisite

### 1. Installation

```Shell
pip install -r requirements.txt
```

### 2. Prepare Dataset

Download [ImageNet](http://www.image-net.org/) Raw image to dir: /path/to/imagenet.  The dir include below folder:

```bash
ls /path/to/imagenet
train  val
```

# Run

### 1. ResNet50

```Shell
cd examples/pytorch/image_recognition/imagenet/cpu/qat
python main.py -t -a resnet50 --pretrained --config /path/to/config_file /path/to/imagenet
```

For ResNet50 model, we can get int8 0.7614 accuracy vs fp32 0.7613.

### 2. ResNet18

```Shell
cd examples/pytorch/image_recognition/imagenet/cpu/qat
python main.py -t -a resnet18 --pretrained --config /path/to/config_file /path/to/imagenet
```

### 3. ResNext101_32x8d

```Shell
cd examples/pytorch/image_recognition/imagenet/cpu/qat
python main.py -t -a resnext101_32x8d --pretrained --config /path/to/config_file /path/to/imagenet
```

Examples Of Enabling LPOT Auto Tuning On PyTorch ResNet
=======================================================

This is a tutorial of how to enable a PyTorch classification model with Intel® Low Precision Optimization Tool.

# User Code Analysis

For quantization aware training mode, Intel® Low Precision Optimization Tool supports two usage as below:

1. User specifies fp32 "model", training function "q_func", evaluation dataset "eval_dataloader" and metric in tuning.metric field of model-specific yaml config file, this option does not require customer to implement evaluation function.
2. User specifies fp32 "model", training function "q_func" and a custom "eval_func" which encapsulates the evaluation dataset and metric by itself, this option require customer implement evaluation function by himself.

As ResNet18/50/101 series are typical classification models, use Top-K as metric which is built-in supported by Intel® Low Precision Optimization Tool. So here we integrate PyTorch ResNet with Intel® Low Precision Optimization Tool by the first use case for simplicity.

### Write Yaml Config File

In examples directory, there is a template.yaml. We could remove most of items and only keep mandotory item for tuning.

```yaml
#conf.yaml

model:
  name: imagenet_qat 
  framework: pytorch

quantization:
  approach: quant_aware_training

evaluation:
  accuracy:
    metric:
      topk: 1

tuning:
    accuracy_criterion:
      relative: 0.01
    exit_policy:
      timeout: 0
    random_seed: 9527

```

Here we choose topk built-in metric and set accuracy target as tolerating 0.01 relative accuracy loss of baseline. The default tuning strategy is basic strategy. The timeout 0 means unlimited tuning time until accuracy target is met, but the result maybe is not a model of best accuracy and performance.

### Prepare

PyTorch quantization requires two manual steps:

1. Add QuantStub and DeQuantStub for all quantizable ops.
2. Fuse possible patterns, such as Conv + Relu and Conv + BN + Relu.

Torchvision provide quantized_model, so we didn't do these steps above for all torchvision models. Please refer [torchvision](https://github.com/pytorch/vision/tree/master/torchvision/models/quantization)

The related code please refer to examples/pytorch/image_recognition/imagenet/cpu/qat/main.py.

### Code Update

After prepare step is done, we just need update main.py like below.

```python
def training_func_for_lpot(model):
    epochs = 8
    iters = 30
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    for nepoch in range(epochs):
        model.train()
        cnt = 0
        for image, target in train_loader:
            print('.', end='')
            cnt += 1
            output = model(image)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if cnt >= iters:
                break
        if nepoch > 3:
            # Freeze quantizer parameters
            model.apply(torch.quantization.disable_observer)
        if nepoch > 2:
            # Freeze batch norm mean and variance estimates
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
    return
model.module.fuse_model()
from lpot.experimental import Quantization, common
quantizer = Quantization("./conf.yaml")
quantizer.model = common.Model(model)
quantizer.q_func = training_func_for_lpot
quantizer.eval_dataloader = val_loader
q_model = quantizer()
```

The quantizer() function will return a best quantized model during timeout constrain.
