# Step by Step
This document describes the step-by-step instructions for pruning ResNet50 on ImageNet dataset. The example refers **pytorch-image-model[](https://github.com/huggingface/pytorch-image-models)**, a popular package for PyTorch image models.

# Prerequisite
## Environment
First, please make sure that you have successfully installed neural_compressor.
```bash
# install dependencies
cd examples/pytorch/image_recognition/ResNet50/pruning/eager/
pip install -r requirements.txt
```
## Prepare Dataset
Download [ImageNet](http://www.image-net.org/) Raw image to dir: /path/to/imagenet.  The dir include below folder:
```bash
ls /path/to/imagenet
train  val
```

# Pruning
Go to the script run_resnet50_prune.sh. Please get familiar with some parameters of pruning by referring to our [Pruning API README](https://github.com/intel/neural-compressor/tree/master/neural_compressor/compression/pruner)
```bash
#!/bin/bash
DATA="/path/to/your/dataset/"
python ./train.py \
    ${DATA} \
    --model "resnet50" \
    --num-classes 1000 \
    --pretrained \
    --batch-size 128 \
    --lr 0.175 \
    --epochs 180 \
    --warmup-epochs 0 \
    --cooldown-epochs 20 \
    --do-prune \
    --do-distillation \
    --target-sparsity 0.75 \
    --pruning-pattern "2x1" \
    --update-frequency-on-step 2000 \
    --distillation-loss-weight "1.0" \
    --output ./path/save/your/models/ \
```
After configs are settled, just run:
```bash
sh run_resnet50_prune.sh
```

If you do not have a GPU, our code will automatically deploy pruning process on CPU. If you do have GPUs and CUDA but you still want to execute the pruning on CPU, use an extra argument of "--no-cuda".
```bash
#!/bin/bash
DATA="/path/to/your/dataset/"
python ./train.py \
    ${DATA} \
    --model "resnet50" \
    --num-classes 1000 \
    --pretrained \
    --batch-size 128 \
    --lr 0.175 \
    --epochs 180 \
    --warmup-epochs 0 \
    --cooldown-epochs 20 \
    --do-prune \
    --do-distillation \
    --target-sparsity 0.75 \
    --pruning-pattern "2x1" \
    --update-frequency-on-step 2000 \
    --distillation-loss-weight "1.0" \
    --output ./path/save/your/models/ \
    --no-cuda
```

# Results
Our dense ResNet50 model's accuracy is 80.1, and our pruned model with 75% 2x1 structured sparsity has accuracy of 78.95.
Your can refer to our validated pruning results in our [documentation](https://github.com/intel/neural-compressor/tree/master/neural_compressor/compression/pruner#validated-pruning-models)

# Distributed Data Parallel Training
We also supported Distributed Data Parallel training on single node and multi nodes settings for pruning. To use Distributed Data Parallel to speedup training, the bash command needs a small adjustment.
<br>
For example, bash command will look like the following, where *`<MASTER_ADDRESS>`* is the address of the master node, it won't be necessary for single node case, *`<NUM_PROCESSES_PER_NODE>`* is the desired processes to use in current node, for node with GPU, usually set to number of GPUs in this node, for node without GPU and use CPU for training, it's recommended set to 1, *`<NUM_NODES>`* is the number of nodes to use, *`<NODE_RANK>`* is the rank of the current node, rank starts from 0 to *`<NUM_NODES>`*`-1`.
<br>
Also please note that to use CPU for training in each node with multi nodes settings, argument `--no_cuda` is mandatory. In multi nodes setting, following command needs to be launched in each node, and all the commands should be the same except for *`<NODE_RANK>`*, which should be integer from 0 to *`<NUM_NODES>`*`-1` assigned to each node.

```bash
DATA="/path/to/your/dataset/"
python -m torch.distributed.launch --master_addr=<MASTER_ADDRESS> --nproc_per_node=<NUM_PROCESSES_PER_NODE> --nnodes=<NUM_NODES> --node_rank=<NODE_RANK> \
    ./train.py \
    ${DATA} \
    --model "resnet50" \
    --num-classes 1000 \
    --pretrained \
    --batch-size 128 \
    --lr 0.175 \
    --epochs 180 \
    --warmup-epochs 0 \
    --cooldown-epochs 20 \
    --do-prune \
    --do-distillation \
    --target-sparsity 0.75 \
    --pruning-pattern "2x1" \
    --update-frequency-on-step 2000 \
    --distillation-loss-weight "1.0" \
    --output ./path/save/your/models/ \
    --no-cuda
    --dist_backend gloo
    --distributed
```

