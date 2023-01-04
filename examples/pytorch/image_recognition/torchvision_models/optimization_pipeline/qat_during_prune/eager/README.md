Step-by-Step
============

This document describes the step-by-step instructions for reproducing PyTorch ResNet50 prune and PTQ results with Intel® Neural Compressor.

# Prerequisite

## 1. Environment
```shell
pip install -r requirements.txt
```

## 2. Prepare Dataset

Download [ImageNet](http://www.image-net.org/) Raw image to dir: /path/to/imagenet.  The dir include below folder:

```bash
ls /path/to/imagenet
train  val
```

# Run

Command is shown as below:

```shell
python -u main.py \
    ~/imagenet \
    --topology resnet50 \
    --prune \
    --quantize \
    --pretrained \
    --pruning_type magnitude \
    --initial_sparsity 0.0 \
    --target_sparsity 0.40 \
    --start_epoch 0 \
    --end_epoch 4 \
    --epochs 5 \
    --output-model saved_results \
    --batch-size 256 \
    --keep-batch-size \
    --lr 0.001
```
