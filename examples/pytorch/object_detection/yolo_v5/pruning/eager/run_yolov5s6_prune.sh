#!/bin/bash

set -x
python3 -m torch.distributed.run --nproc_per_node 2 --master_port='29500' \
        examples/pruning/yolov5/train_prune.py \
        --data examples/pruning/yolov5/data/coco.yaml \
        --hyp examples/pruning/yolov5/data/hyp.scratch-low.yaml \
        --pruning_config "./examples/pruning/yolov5/data/yolov5s6_prune.yaml" \
        --weights ./examples/pruning/yolov5/teacher_weights/yolov5s6.pt \
        --device 4,6 \
        --img 640 \
        --do_prune \
        --epochs 250 \
        --cooldown_epochs 150 \
        --batch-size 64 \
        --patience 0
