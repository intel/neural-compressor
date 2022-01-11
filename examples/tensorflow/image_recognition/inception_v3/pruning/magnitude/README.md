Step-by-Step
============

This document is used to list steps of reproducing Intel® Neural Compressor magnitude pruning feature on Inception-V3 model.


## Prerequisite

### 1. Install Intel® Neural Compressor
```shell
pip install neural-compressor
```
### 2. Install other requirements
```shell
pip install -r requirements.txt
```

## Run command to prune the model
Run the command to get baseline model and then prune it and save the pruned model into './Inception-V3_Model'.
```shell
python main.py 
```
If you want to accelerate pruning with multi-node distributed training and evaluation, you only need to add a small amount of code and use horovod to run main.py. As shown in main.py, uncomment two lines 'prune.train_distributed = True' and 'prune.evaluation_distributed = True' in main.py is all you need. Run the command to get pruned model with multi-node distributed training and evaluation.
```shell
horovodrun -np <num_of_processes> -H <hosts> python main.py
```
