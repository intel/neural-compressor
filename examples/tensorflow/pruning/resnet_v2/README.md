Step-by-Step
============

This document is used to list steps of reproducing Intel® Low Precision Optimization Tool magnitude pruning feature.


## Prerequisite

### 1. Installation
```shell
# Install Intel® Low Precision Optimization Tool
pip install lpot
```
### 2. Install Intel Tensorflow 2.4.0 or above.
```shell
pip install intel-tensorflow==2.4.0
```

## Run Command
  ```shell
  python main.py     # to get pruned model which saved into './pruned_model'.

  python benchmark.py   # to run performance benchmark.
  ```

