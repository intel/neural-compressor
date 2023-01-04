Step-by-Step
============

This document describes the step-by-step instructions for reproducing PyTorch pruning results with Intel® Neural Compressor.

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
### Non-distributed
**run_pruning_cpu.sh** is an example.
```shell
python -u main.py \
    /path/to/imagenet/ \
    --topology resnet18 \
    --prune \
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
    --lr 0.001 \
    --iteration 30 \
```

### Distributed 
**run_pruning_distributed_cpu.sh** is an example.
```shell
horovodrun -np 2 python -u main.py \
    /path/to/imagenet/ \
    --topology resnet18 \
    --prune \
    --pretrained \
    --pruning_type magnitude \
    --initial_sparsity 0.0 \
    --target_sparsity 0.40 \
    --start_epoch 0 \
    --end_epoch 9 \
    --epochs 10 \
    --output-model saved_results \
    --world-size 1 \
    --num-per-node 2 \
    --batch-size 256 \
    --keep-batch-size \
    --lr 0.001 \
    --iteration 30 \
```

### Other notes
- Topology supports resnet18/resnet34/resnet50/resnet101
- World-size and num-per-node should match to np of horovodrun. For example as run_pruning_distributed_cpu.sh, np of horovodrun is 2, and world-size * num-per-node = 1 * 2 = 2.
