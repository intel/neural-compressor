Step-by-Step
============

This document describes the step-by-step instructions for reproducing PyTorch ResNet18 MixedPrecision results with Intel® Neural Compressor.

# Prerequisite

### 1. Environment

PyTorch 1.8 or higher version is needed with pytorch_fx backend.

```Shell
cd examples/pytorch/image_recognition/torchvision_models/mixed_precision/resnet18
pip install -r requirements.txt
```
> Note: Validated PyTorch [Version](/docs/source/installation_guide.md#validated-software-environment).

### 2. Prepare Dataset

Download [ImageNet](http://www.image-net.org/) Raw image to dir: /path/to/imagenet. The dir includes below folder:

```bash
ls /path/to/imagenet
train  val
```

# Run

> Note: All torchvision model names can be passed as long as they are included in `torchvision.models`, below are some examples.

## MixedPrecision
```Shell
python main.py -t -a resnet18 --pretrained /path/to/imagenet
```

## Benchmark
```Shell
# run optimized performance
bash run_benchmark.sh --input_model=resnet18 --dataset_location=/path/to/imagenet --mode=performance --batch_size=1 --optimized=true --iters=500
# run optimized accuracy
bash run_benchmark.sh --input_model=resnet18 --dataset_location=/path/to/imagenet --mode=accuracy --batch_size=100 --optimized=true
```





