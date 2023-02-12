Step-by-Step
============

This document is used to list steps of reproducing TensorFlow keras model resnet50_fashion quantization and benchmark using Intel® Neural Compressor.
This example can run on Intel CPUs and GPUs.


# Prerequisite

## 1. Environment

### Installation
```shell
# Install Intel® Neural Compressor
pip install neural-compressor
```

### Install Intel Tensorflow
```shell
pip install intel-tensorflow
```

> Note: Validated TensorFlow [Version](/docs/source/installation_guide.md#validated-software-environment).

### Install Intel Extension for Tensorflow
#### Quantizing the model on Intel GPU
Intel Extension for Tensorflow is mandatory to be installed for quantizing the model on Intel GPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[gpu]
```
For any more details, please follow the procedure in [install-gpu-drivers](https://github.com/intel-innersource/frameworks.ai.infrastructure.intel-extension-for-tensorflow.intel-extension-for-tensorflow/blob/master/docs/install/install_for_gpu.md#install-gpu-drivers)

#### Quantizing the model on Intel CPU(Experimental)
Intel Extension for Tensorflow for Intel CPUs is experimental currently. It's not mandatory for quantizing the model on Intel CPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[cpu]
```

## 2. Prepare Pretrained model

Run the `resnet50_fashion_mnist_train.py` script located in `examples/tensorflow/image_recognition/keras_models/resnet50_fashion/quantization/ptq`, and it will generate a saved model called `resnet50_fashion` at current path.

## 3. Prepare dataset

Please download FashionMNIST dataset(https://github.com/zalandoresearch/fashion-mnist) in advance.


# Run Command
## 1 Quantization
  ```shell
  bash run_tuning.sh --input_model=./resnet50_fashion --output_model=./result --dataset_location=/path/to/FashionMNIST/
  ```

## 2. Benchmark
  ```shell
  bash run_benchmark.sh --input_model=./result --mode=accuracy --dataset_location=/path/to/FashionMNIST/ --batch_size=32
  bash run_benchmark.sh --input_model=./result --mode=performance --dataset_location=/path/to/FashionMNIST/ --batch_size=1
  ```
