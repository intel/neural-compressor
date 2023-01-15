Step-by-Step
============
This document describes the step-by-step instructions for reproducing the distillation on CIFAR10 dataset. The example is a distillation of the WideResNet40-2 to the MobileNetV2-0.35.

# Prerequisite
## Environment
```shell
# install dependencies
cd examples/pytorch/image_recognition/MobileNetV2-0.35/distillation/eager
pip install -r requirements.txt
```

# Distillation
Distillation examples on CIFAR10.

## 1. Train Teacher Model
```shell
# for training of the teacher model WideResNet40-2
python train_without_distillation.py --epochs 200 --lr 0.1 --layers 40 --widen-factor 2 --name WideResNet-40-2 --tensorboard
```

## 2. Distilling The Teacher Model to The Student Model
```shell
# for distillation of the teacher model WideResNet40-2 to the student model MobileNetV2-0.35
python main.py --epochs 200 --lr 0.02 --name MobileNetV2-0.35-distillation --teacher_model runs/WideResNet-40-2/model_best.pth.tar --tensorboard --seed 9 
```

## Distributed Data Parallel Training
We also supported Distributed Data Parallel training on single node and multi nodes settings for distillation. To use Distributed Data Parallel to speedup training, the bash command needs a small adjustment.
<br>
For example, bash command will look like the following, where *`<MASTER_ADDRESS>`* is the address of the master node, it won't be necessary for single node case, *`<NUM_PROCESSES_PER_NODE>`* is the desired processes to use in current node, for node with GPU, usually set to number of GPUs in this node, for node without GPU and use CPU for training, it's recommended set to 1, *`<NUM_NODES>`* is the number of nodes to use, *`<NODE_RANK>`* is the rank of the current node, rank starts from 0 to *`<NUM_NODES>`*`-1`.
<br>
Also please note that to use CPU for training in each node with multi nodes settings, argument `--no_cuda` is mandatory. In multi nodes setting, following command needs to be launched in each node, and all the commands should be the same except for *`<NODE_RANK>`*, which should be integer from 0 to *`<NUM_NODES>`*`-1` assigned to each node.

```bash
python -m torch.distributed.launch --master_addr=<MASTER_ADDRESS> --nproc_per_node=<NUM_PROCESSES_PER_NODE> --nnodes=<NUM_NODES> --node_rank=<NODE_RANK> \
   main.py --epochs 200 --lr 0.02 --name MobileNetV2-0.35-distillation --teacher_model runs/WideResNet-40-2/model_best.pth.tar --tensorboard --seed 9
```