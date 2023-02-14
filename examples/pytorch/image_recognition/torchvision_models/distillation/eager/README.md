Step-by-Step
============

This is an example to show the usage of distillation.

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
bash run_distillation.sh --topology=(resnet18|resnet34|resnet50|resnet101) --teacher=(resnet18|resnet34|resnet50|resnet101)  --dataset_location=/path/to/imagenet --output_model=path/to/output_model
```

> Note: `--topology` is the student model and `--teacher` is the teacher model.

We also supported Distributed Data Parallel training on single node and multi nodes settings for distillation. To use Distributed Data Parallel to speedup training, the bash command needs a small adjustment.
<br>
For example, bash command will look like the following, where *`<MASTER_ADDRESS>`* is the address of the master node, it won't be necessary for single node case, *`<NUM_PROCESSES_PER_NODE>`* is the desired processes to use in current node, for node with GPU, usually set to number of GPUs in this node, for node without GPU and use CPU for training, it's recommended set to 1, *`<NUM_NODES>`* is the number of nodes to use, *`<NODE_RANK>`* is the rank of the current node, rank starts from 0 to *`<NUM_NODES>`*`-1`.
<br>
Also please note that to use CPU for training in each node with multi nodes settings, argument `--no_cuda` is mandatory. In multi nodes setting, following command needs to be launched in each node, and all the commands should be the same except for *`<NODE_RANK>`*, which should be integer from 0 to *`<NUM_NODES>`*`-1` assigned to each node.

```bash
python -m torch.distributed.launch --master_addr=<MASTER_ADDRESS> --nproc_per_node=<NUM_PROCESSES_PER_NODE> --nnodes=<NUM_NODES> --node_rank=<NODE_RANK> \
 main.py --topology=(resnet18|resnet34|resnet50|resnet101) --teacher=(resnet18|resnet34|resnet50|resnet101) \
  --dataset_location=/path/to/imagenet --output_model=path/to/output_model --distillation --pretrained --no_cuda
```
# Results
We ran distillation for ResNet50 on ImageNet dataset with several settings, teacher model set to ResNet101, results are shown below.
|   Setting       | Top1 Accuracy  |  Elapsed Time |
|-----------------|----------------|---------------|
|   1 ICX8360Y Machine     |   76.73%       |        2765min  |
|   4 ICX8360Y Machines     |   76.63%       |        1076min  |