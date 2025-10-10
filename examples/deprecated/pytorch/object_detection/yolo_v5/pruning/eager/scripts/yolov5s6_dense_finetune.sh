#!/bin/bash
set -x

    python3 -m torch.distributed.run --nproc_per_node 2 --master_port='29500' \
        examples/pytorch/object_detection/yolo_v5/pruning/eager/train.py \
        --data "./coco.yaml" \
        --hyp "./hyp.scratch-low.yaml" \
        --weights "/path/to/dense_pretrained_model/" \
        --device 0,1 \
        --img 640 \
        --epochs 300 \
        --batch-size 64