Step-by-Step
============

This document is used to list steps of reproducing Intel® Neural Compressor magnitude pruning feature on ViT model.


# Prerequisite

## 1. Environment

### Install Intel® Neural Compressor
```shell
pip install neural-compressor
```
### Install requirements
```shell
pip install -r requirements.txt
```

## 2. Prepare Model
Run the script to save a baseline model to the directory './ViT_Model'.
```python
python prepare_model.py
```

# Run
Run the command to prune the baseline model and save it into a given path.
The CIFAR100 dataset will be automatically loaded.

```shell
python main.py --output_model=/path/to/output_model/
```

If you want to accelerate pruning with multi-node distributed training and evaluation, you only need to add twp arguments and use horovod to run main.py.  Run the command to get pruned model with multi-node distributed training and evaluation.

```shell
horovodrun -np <num_of_processes> -H <hosts> python main.py --output_model=/path/to/output_model/ --train_distributed --evaluation_distributed
```