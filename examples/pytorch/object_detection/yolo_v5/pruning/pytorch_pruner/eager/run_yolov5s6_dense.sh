#!/bin/bash

set -x
python3 -m torch.distributed.run --nproc_per_node 4 --master_port='29500' \
        examples/pruning/yolov5/train_prune.py \
        --data examples/pruning/yolov5/data/coco.yaml \
        --hyp examples/pruning/yolov5/data/hyp.scratch-low.yaml \
	--pruning_config examples/pruning/yolov5/data/yolov5s6_prune.yaml \
	--weights ./examples/pruning/yolov5/yolov5s6.pt \
        --device 0,1,2,3 \
        --img 640 \
        --epochs 300 \
        --batch-size 128
