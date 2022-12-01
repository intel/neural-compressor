#!/bin/bash

set -x
python3 examples/pruning/yolov5/val.py \
        --data ./examples/pruning/yolov5/data/coco.yaml \
        --weights examples/pruning/yolov5/runs/train/exp/weights/best.pt \
        --batch-size 32

