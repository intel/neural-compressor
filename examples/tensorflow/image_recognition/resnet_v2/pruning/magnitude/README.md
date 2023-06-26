Step-by-Step
============

This document is used to list steps of reproducing Intel® Neural Compressor magnitude pruning feature.


# Prerequisite

## 1. Environment

###  Install Intel® Neural Compressor
```shell
pip install neural-compressor
```
### Install TensorFlow
```shell
pip install tensorflow
```

# Run
Run the command to get pretrained baseline model which will be saved to './baseline_model'. Then, the model will be pruned and saved into a given path.
The CIFAR10 dataset will be automatically loaded.
```shell
python main.py --output_model=/path/to/output_model/ --prune
```
If you want to accelerate pruning with multi-node distributed training and evaluation, you only need to add two arguments and use horovod to run main.py.
Use horovod to run main.py to get pruned model with multi-node distributed training and evaluation.
```shell
horovodrun -np <num_of_processes> -H <hosts> python main.py --output_model=/path/to/output_model/  --train_distributed --evaluation_distributed  --prune
```

Run the command to get pruned model performance.
```shell
python main.py --input_model=/path/to/input_model/ --benchmark
```
