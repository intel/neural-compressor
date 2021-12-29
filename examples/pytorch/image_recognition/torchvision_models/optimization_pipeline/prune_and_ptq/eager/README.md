Step-by-Step
============

This document describes the step-by-step instructions for reproducing PyTorch ResNet50 prune and PTQ results with Intel® Neural Compressor.

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
cd examples/pytorch/image_recognition/torchvision_models/optimization_pipeline/prune_and_ptq/eager
```

#### Run prune and PTQ
```
python main.py -t --prune -a resnet50 --pretrained /path/to/imagenet

```

#### Run prune only
```
python main.py --prune -a resnet50 --pretrained /path/to/imagenet

```

#### Run PTQ only
```
python main.py -t -a resnet50 --pretrained /path/to/imagenet

```

### 4. Scheduler

In examples directory, there are two yaml templates `prune_conf.yaml` and `ptq_conf.yaml` which are used in pruning and post training quantization. User could some of the items in yaml and only keep mandatory item.

Neural Compressor defined Scheduler to do both prune and PTQ in one turn. It is sufficient to add following lines of code to execute pruning and PTQ in scheduler.
```
prune = Pruning('./prune_conf.yaml')
quantizer = Quantization('./ptq_conf.yaml')
scheduler = Scheduler()
scheduler.model = common.Model(model)
scheduler.append(prune)
scheduler.append(quantizer)
opt_model = scheduler()
```


