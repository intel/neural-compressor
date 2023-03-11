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
    --epochs 60 \
    --warmup-epochs 0 \
    --cooldown-epochs 10 \
    --do-prune \
    --do-distillation \
    --target-sparsity 0.75 \
    --pruning-pattern "2x1" \
    --update-frequency-on-step 2000 \
    --distillation-loss-weight "1.0" \
    --output ./outputs/ \

```
After configs are settled, just run:
```bash
sh run_resnet50_prune.sh
```

# Results
Our dense ResNet50 model's accuracy is 80.1, and our pruned model with 75% 2x1 structured sparsity has accuracy of 78.95.
Your can refer to our validated pruning results in our [documentation](https://github.com/intel/neural-compressor/tree/master/neural_compressor/compression/pruner#validated-pruning-models)
