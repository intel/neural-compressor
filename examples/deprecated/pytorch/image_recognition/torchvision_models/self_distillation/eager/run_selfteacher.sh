#!/bin/bash

echo "Run Self Teacher Distillation on CIFAR100"
bash run_distillation.sh --topology=resnet50 --output_model=/path/to/output_model --dataset_location=/path/to/CIFAR100_root/ --use_cpu=1
