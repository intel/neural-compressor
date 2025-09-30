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
    --output ./path/to/your/models/ \
