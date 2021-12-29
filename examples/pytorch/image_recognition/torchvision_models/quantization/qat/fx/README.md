Step-by-Step
============

This document describes the step-by-step instructions for reproducing PyTorch ResNet50/ResNet18/ResNet101 tuning results with Intel® Neural Compressor.

# Prerequisite

### 1. Installation

PyTorch 1.8 or higher version is needed with pytorch_fx backend.

```Shell
cd examples/pytorch/image_recognition/torchvision_models/quantization/qat/fx
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

Examples Of Enabling Neural Compressor Auto Tuning On PyTorch ResNet
=======================================================

This is a tutorial of how to enable a PyTorch classification model with Intel® Neural Compressor.

# User Code Analysis

For quantization aware training mode, Intel® Neural Compressor supports two usage as below:

1. User specifies fp32 "model", training function "q_func", evaluation dataset "eval_dataloader" and metric in tuning.metric field of model-specific yaml config file, this option does not require customer to implement evaluation function.
2. User specifies fp32 "model", training function "q_func" and a custom "eval_func" which encapsulates the evaluation dataset and metric by itself, this option require customer implement evaluation function by himself.

As ResNet18/50/101 series are typical classification models, use Top-K as metric which is built-in supported by Intel® Neural Compressor. So here we integrate PyTorch ResNet with Intel® Neural Compressor by the first use case for simplicity.

### Write Yaml Config File

In examples directory, there is a template.yaml. We could remove most of the items and only keep mandatory item for tuning.

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

The related code please refer to examples/pytorch/image_recognition/torchvision_models/quantization/qat/fx/main.py.

### Code Update

After prepare step is done, we just need update main.py like below.

```python
def training_func_for_nc(model):
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
    return model
model.module.fuse_model()
from neural_compressor.experimental import Quantization, common
quantizer = Quantization("./conf.yaml")
quantizer.model = common.Model(model)
quantizer.q_func = training_func_for_nc
quantizer.eval_dataloader = val_loader
q_model = quantizer.fit()
```

The quantizer.fit() function will return a best quantized model during timeout constrain.
