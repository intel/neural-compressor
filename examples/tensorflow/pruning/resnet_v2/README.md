Step-by-Step
============

This document is used to list steps of reproducing Intel® Neural Compressor magnitude pruning feature.


## Prerequisite

### 1. Installation
```shell
# Install Intel® Neural Compressor
pip install neural-compressor
```
### 2. Install Intel Tensorflow 2.4.0 or above.
```shell
pip install intel-tensorflow==2.4.0
```

## Run Command
  ```shell
  python main.py     # to get pruned model which overwritten and saved into './baseline_model'.
  
  # If you want to accelerate pruning with multi-node distributed training and evaluation, you only need to add a small amount of code and use horovod to run main.py.
  # As shown in main.py, add two lines 'prune.train_distributed = True' and 'prune.evaluation_distributed = True' into main.py.
  horovodrun -np <num_of_processes> -H <hosts> python main.py    # run main.py to get pruned model with multi-node distributed training and evaluation.

  python benchmark.py   # to run performance benchmark.
  ```

