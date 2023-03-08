#!/bin/bash
export CUDA_VISIBLE_DEVICES="6"
DATA="/dataset/imagenet/img_raw/"
/home/cyy/anaconda3/envs/cyy_resnet50/bin/python ./examples/pytorch/image_recognition/ResNet50/pruning/eager/train.py \
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
    --output ./outputs/03-08/ \
    2>&1 | tee ./outputs/03-08/03-08.log
