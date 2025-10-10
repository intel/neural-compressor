Step-by-Step
============
This document describes the step-by-step instructions for reproducing the distillation on CIFAR100 dataset. The example is a distillation of the CNN-10 to the CNN-2.

# Prerequisite
## Environment
```shell
# install dependencies
cd examples/pytorch/image_recognition/CNN-2/distillation/eager
pip install -r requirements.txt
```
> Note: Validated PyTorch [Version](/docs/source/installation_guide.md#validated-software-environment).

# Distillation
Distillation examples on CIFAR100.

## 1. Train Teacher Model
```shell
# for training of the teacher model CNN-10
python train_without_distillation.py --model_type CNN-10 --epochs 200 --lr 0.1 --tensorboard
```

## 2. Distilling The Student Model with The Teacher Model
```shell
# for distillation of the student model CNN-2 with the teacher model CNN-10
python main.py --epochs 200 --lr 0.02 --name CNN-2-distillation --student_type CNN-2 --teacher_type CNN-10 --teacher_model runs/CNN-10/model_best.pth.tar --tensorboard
```

## 3. Distributed Data Parallel Training
We also supported Distributed Data Parallel training on single node and multi nodes settings for distillation. To use Distributed Data Parallel to speedup training, the bash command needs a small adjustment.
<br>
For example, bash command will look like the following, where *`<MASTER_ADDRESS>`* is the address of the master node, it won't be necessary for single node case, *`<NUM_PROCESSES_PER_NODE>`* is the desired processes to use in current node, for node with GPU, usually set to number of GPUs in this node, for node without GPU and use CPU for training, it's recommended set to 1, *`<NUM_NODES>`* is the number of nodes to use, *`<NODE_RANK>`* is the rank of the current node, rank starts from 0 to *`<NUM_NODES>`*`-1`.
<br>
Also please note that to use CPU for training in each node with multi nodes settings, argument `--no_cuda` is mandatory. In multi nodes setting, following command needs to be launched in each node, and all the commands should be the same except for *`<NODE_RANK>`*, which should be integer from 0 to *`<NUM_NODES>`*`-1` assigned to each node.

```bash
python -m torch.distributed.launch --master_addr=<MASTER_ADDRESS> --nproc_per_node=<NUM_PROCESSES_PER_NODE> --nnodes=<NUM_NODES> --node_rank=<NODE_RANK> \
   main.py --epochs 200 --lr 0.02 --name CNN-2-distillation --student_type CNN-2 --teacher_type CNN-10 --teacher_model runs/CNN-10/model_best.pth.tar --tensorboard
```