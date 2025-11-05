## YOLOv5's Version
YOLOv5 pruned example is developed based on Version 6.2 [YOLOv5](https://github.com/ultralytics/yolov5/releases/tag/v6.2), for unmodified code files such as 'val', 'models' and 'utils', please get them from the YOLOv5 repository.


## Examples
Several pruning examples are provided, which are trained on different datasets/tasks, use different sparsity patterns, etc. We are working on sharing our sparse models on HuggingFace.

There are [Pruning Scripts](https://github.com/intel/neural-compressor/tree/master/examples/pytorch/object_detection/yolo_v5/pruning/eager/scripts/) for YOLOv5s sparse models. The sparse models with different patterns ("4x1", "1x1", etc) can be obtained by modifying "target_sparsity" and "pruning_pattern" parameters.

Dense models can also be fine-tuned on COCO datasets (by setting --do_prune to False) [YOLOv5s6-COCO](https://github.com/intel/neural-compressor/tree/master/examples/pytorch/object_detection/yolo_v5/pruning/eager/scripts/yolov5s6_dense_finetune.sh)


#### YOLOv5
The snip-momentum pruning method is used by default and the initial dense models are all fine-tuned.
|  Model  | Dataset  |  Sparsity pattern |Element-wise/matmul, GEMM, Conv ratio | Dense mAP50/mAP50-95 | Sparse mAP50/mAP50-95| Relative drop|
|  :----:  | :----:  | :----: | :----: |:----:|:----:| :----: |
| YOLOv5s6 | COCO |  1x1  | 0.7998 | 0.600:0.404 | 0.584/0.393 | -2.67%/-2.71% |
| YOLOv5s6 | COCO |  4x1  | 0.7776 | 0.600:0.404 | 0.573/0.381 | -4.50%/-5.69% |


## References
* [SNIP: Single-shot Network Pruning based on Connection Sensitivity](https://arxiv.org/abs/1810.02340)
* [Object detection at 200 Frames Per Second](https://arxiv.org/pdf/1805.06361.pdf)


