Step-by-Step
============

This document describes the step-by-step instructions for reproducing PyTorch ResNet50 prune and QAT results with Intel® Neural Compressor.

> **Note**
>
> * PyTorch quantization implementation in imperative path has limitation on automatically execution. It requires to manually add QuantStub and DequantStub for quantizable ops, it also requires to manually do fusion operation.
> * Neural Compressor supposes user have done these two steps before invoking Neural Compressor interface.
>   For details, please refer to https://pytorch.org/docs/stable/quantization.html

# Prerequisite

### 1. Installation

```shell
pip install -r requirements.txt
```

### 2. Prepare Dataset

Download [ImageNet](http://www.image-net.org/) Raw image to dir: /path/to/imagenet.  The dir include below folder:

```bash
ls /path/to/imagenet
train  val
```

### 3. Run


```shell
cd examples/pytorch/image_recognition/torchvision_models/optimization_pipeline/qat_during_prune/eager
```

#### Run qat during prune
```
python main.py -t -a resnet50 --pretrained /path/to/imagenet
```

### 4. Scheduler

In examples directory, there are two yaml templates `prune_conf.yaml` and `qat_conf.yaml` which are used in pruning and quantization aware training. User could some of the items in yaml and only keep mandatory item.

Neural Compressor defined Scheduler to do QAT during prune in one turn. It is sufficient to add following lines of code to execute pruning and QAT in scheduler.
```
quantizer = Quantization('./qat_conf.yaml')
prune = Pruning('./prune_conf.yaml')
scheduler = Scheduler()
scheduler.model = common.Model(model)
combination = scheduler.combine(prune, quantizer)
print(combination)
scheduler.append(combination)
opt_model = scheduler()
```


