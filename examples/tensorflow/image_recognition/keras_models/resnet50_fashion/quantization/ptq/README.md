Step-by-Step
============

This document is used to list steps of reproducing TensorFlow keras Intel® Neural Compressor tuning zoo result.
This example can run on Intel CPUs and GPUs.


## Prerequisite

### 1. Installation
```shell
# Install Intel® Neural Compressor
pip install neural-compressor
```
### 2. Install Intel Tensorflow
```shell
pip install intel-tensorflow
```
> Note: Supported Tensorflow [Version](../../../../../../../README.md#supported-frameworks).

### 3. Install Intel Extension for Tensorflow
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

### 4. Prepare Pretrained model

Run the `resnet50_fashion_mnist_train.py` script located in `LowPrecisionInferenceTool/examples/tensorflow/keras`, and it will generate a saved model called `resnet50_fashion` at current path.

### 5. Prepare dataset

If users download FashionMNIST dataset in advance, please set dataset part in yaml as follows:

```yaml
dataset:
  FashionMNIST:
    root: /path/to/user/dataset
```

Otherwise, if users do not download dataset in advance, please set the dataset root to any place which can be accessed and it will automatically download FashionMNIST dataset to the corresponding path.

## Write Yaml config file
In examples directory, there is a resnet50_fashion.yaml for tuning the model on Intel CPUs. The 'framework' in the yaml is set to 'tensorflow'. If running this example on Intel GPUs, the 'framework' should be set to 'tensorflow_itex' and the device in yaml file should be set to 'gpu'. The resnet50_fashion_itex.yaml is prepared for the GPU case. We could remove most of items and only keep mandatory item for tuning. We also implement a calibration dataloader and have evaluation field for creation of evaluation function at internal neural_compressor.

## Run Command
  ```shell
  bash run_tuning.sh --config=resnet50_fashion.yaml --input_model=./resnet50_fashion --output_model=./result
  bash run_benchmark.sh --config=resnet50_fashion.yaml --input_model=./resnet50_fashion --mode=performance
  ```
