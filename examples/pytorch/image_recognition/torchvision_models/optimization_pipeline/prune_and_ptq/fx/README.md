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

```python
python -u main.py \
    /path/to/imagenet \
    --arch resnet50 \
    --prune \
    --quantize \
    --pretrained \
    --pruning_type magnitude \
    --initial_sparsity 0.0 \
    --target_sparsity 0.40 \
    --start_epoch 0 \
    --end_epoch 9 \
    --epochs 10 \
    --output-model saved_results \
    --batch-size 256 \
    --keep-batch-size \
    --lr 0.001
```

Please get familiar with some parameters of pruning by referring to our [Pruning API README](https://github.com/intel/neural-compressor/tree/master/neural_compressor/compression/pruner)
