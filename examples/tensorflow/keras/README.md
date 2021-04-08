Step-by-Step
============

This document is used to list steps of reproducing TensorFlow keras Intel® Low Precision Optimization Tool tuning zoo result.


## Prerequisite

### 1. Installation
```shell
# Install Intel® Low Precision Optimization Tool
pip instal lpot
```
### 2. Install Intel Tensorflow
```shell
pip install intel-tensorflow
```
> Note: Supported Tensorflow [Version](../../../README.md).

### 3. Prepare Pretrained model

Run the `resnet50_fashion_mnist_train.py` script located in `LowPrecisionInferenceTool/examples/tensorflow/keras`, and it will generate a saved model called `resnet50_fashion` at current path.


## Run Command
  ```shell
  bash run_tuning.sh --config=resnet50_fashion.yaml --input_model=./resnet50_fashion --output_model=./result
  bash run_benchmark.sh --config=resnet50_fashion.yaml --input_model=./resnet50_fashion
  ```

