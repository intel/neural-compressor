#!/bin/bash
set -x

    python3 -m torch.distributed.run --nproc_per_node 1 --master_port='29500' \
        examples/pytorch/object_detection/yolo_v5/pruning/eager/train.py \
        --data "./coco.yaml" \
        --hyp "./hyp.scratch-low.yaml" \
        --weights "/path/to/yolov5/dense_finetuned_model/" \
        --device 0,1 \
        --img 640 \
        --do_prune \
        --epochs 250 \
        --cooldown_epochs 150 \
        --batch-size 64 \
        --patience 0 \
        --target_sparsity 0.8 \
        --pruning_pattern "4x1" \
        --pruning_frequency 1000