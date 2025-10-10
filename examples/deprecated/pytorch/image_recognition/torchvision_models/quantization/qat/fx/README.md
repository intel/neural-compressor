Step-by-Step
============

This document describes the step-by-step instructions for reproducing PyTorch ResNet50/ResNet18/ResNet101 tuning results with Intel® Neural Compressor.

# Prerequisite

### 1. Environment

PyTorch 1.8 or higher version is needed with pytorch_fx backend.

```Shell
cd examples/pytorch/image_recognition/torchvision_models/quantization/qat/fx
pip install -r requirements.txt
```
> Note: Validated PyTorch [Version](/docs/source/installation_guide.md#validated-software-environment).

### 2. Prepare Dataset

Download [ImageNet](http://www.image-net.org/) Raw image to dir: /path/to/imagenet.  The dir include below folder:

```bash
ls /path/to/imagenet
train  val
```

# Run

> Note: All torchvision model names can be passed as long as they are included in `torchvision.models`, below are some examples.

### 1. ResNet50

```Shell
python main.py -t -a resnet50 --pretrained /path/to/imagenet
```

### 2. ResNet18

```Shell
python main.py -t -a resnet18 --pretrained /path/to/imagenet
```

### 3. ResNext101_32x8d

```Shell
python main.py -t -a resnext101_32x8d --pretrained /path/to/imagenet
```

# Distributed Data Parallel Training
We also supported Distributed Data Parallel training on single node and multi nodes settings for QAT. To use Distributed Data Parallel to speedup training, the bash command needs a small adjustment.
<br>
For example, bash command will look like the following, where *`<MASTER_ADDRESS>`* is the address of the master node, it won't be necessary for single node case, *`<NUM_PROCESSES_PER_NODE>`* is the desired processes to use in current node, for node with GPU, usually set to number of GPUs in this node, for node without GPU and use CPU for training, it's recommended set to 1, *`<NUM_NODES>`* is the number of nodes to use, *`<NODE_RANK>`* is the rank of the current node, rank starts from 0 to *`<NUM_NODES>`*`-1`.
<br>
Also please note that to use CPU for training in each node with multi nodes settings, argument `--no_cuda` is mandatory. In multi nodes setting, following command needs to be launched in each node, and all the commands should be the same except for *`<NODE_RANK>`*, which should be integer from 0 to *`<NUM_NODES>`*`-1` assigned to each node.

```bash
python -m torch.distributed.launch --master_addr=<MASTER_ADDRESS> --nproc_per_node=<NUM_PROCESSES_PER_NODE> --nnodes=<NUM_NODES> --node_rank=<NODE_RANK> \
   main.py -t -a resnet50 --pretrained /path/to/imagenet
```

# Results
We ran QAT for ResNet50 on ImageNet dataset with several settings, results are shown below.
|   Setting       | Top1 Accuracy  |  Elapsed Time |
|-----------------|----------------|---------------|
|   1 CLX6248 Machine     |   76.22%       |        55min  |
|   2 CLX6248 Machines     |   76.15%       |        26min  |
|   4 CLX6248 Machines     |   75.99%       |        15min  |
