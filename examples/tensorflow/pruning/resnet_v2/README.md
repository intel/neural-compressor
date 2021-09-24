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
  python main.py     # to get pruned model which saved into './pruned_model'.

  python benchmark.py   # to run performance benchmark.
  ```

