## YOLOv5's Version
YOLOv5 pruned example is developed based on Version 6.2[YOLOv5]https://github.com/ultralytics/yolov5/releases/tag/v6.2

## Examples
we have provided several pruning examples, which are trained on different datasets/tasks, use different sparsity patterns, etc. We are working on sharing our sparse models on HuggingFace.
### [YOLOv5](https://github.com/intel/neural-compressor/tree/master/examples/pytorch/object_detection/yolo_v5/pruning/pytorch_pruner/eager)

We can train a sparse model with NxM (1x1/4x1) pattern:
```shell
python3 -m torch.distributed.run --nproc_per_node 2 --master_port='29500' \
        ./train.py \
        --data "./coco.yaml" \
        --hyp "./hyp.scratch-low.yaml" \
        --weights "/path/to/yolov5s/dense_finetuned_model/" \
        --device 0,1 \
        --img 640 \
        --do_prune \
        --epochs 250 \
        --cooldown_epochs 150 \
        --batch-size 64 \
        --patience 0 \
        --target_sparsity 0.8 \
        --pruning_pattern "1x1" \
        --pruning_frequency 1000
```

We can also choose pruning with distillation(l2/kl):
```shell
python3 -m torch.distributed.run --nproc_per_node 2 --master_port='29500' \
        ./train.py \
        --data "./coco.yaml" \
        --hyp "./hyp.scratch-low.yaml" \
        --weights "/path/to/yolov5/dense_finetuned_model/" \
        --device 0,1 \
        --img 640 \
        --do_prune \
        --dist_loss l2 \
        --temperature 10 \
        --epochs 250 \
        --cooldown_epochs 150 \
        --batch-size 64 \
        --patience 0 \
        --target_sparsity 0.8 \
        --pruning_pattern "4x1" \
        --pruning_frequency 1000
```

Dense model training is also supported as following (by setting --do_prune to False):
```shell
python3 -m torch.distributed.run --nproc_per_node 2 --master_port='29500' \
        ./train.py \
        --data "./coco.yaml" \
        --hyp "./hyp.scratch-low.yaml" \
        --weights "/path/to/dense_pretrained_model/" \
        --device 0,1 \
        --img 640 \
        --epochs 300 \
        --batch-size 64
```

#### YOLOv5
The snip-momentum pruning method is used by default and the initial dense models are all fine-tuned.
|  Model  | Dataset  |  Sparsity pattern |Element-wise/matmul, GEMM, Conv ratio | Dense mAP50/mAP50-95 | Sparse mAP50/mAP50-95| Relative drop|
|  :----:  | :----:  | :----: | :----: |:----:|:----:| :----: |
| YOLOv5s6 | COCO |  1x1  | 0.7998 | 0.600:0.404 | 0.584/0.393 | -2.67%/-2.71% |
| YOLOv5s6 | COCO |  4x1  | 0.7776 | 0.600:0.404 | 0.573/0.381 | -4.50%/-5.69% |

## References
* [SNIP: Single-shot Network Pruning based on Connection Sensitivity](https://arxiv.org/abs/1810.02340)
* [Object detection at 200 Frames Per Second](https://arxiv.org/pdf/1805.06361.pdf)


