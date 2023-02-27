Step-by-Step
============

This document list steps of demo the usage of tensorflow distillation via Neural Compressor.
This example can run on Intel CPUs and GPUs.

> **Note**: 
> Most of those models are both supported in Intel optimized TF 1.15.x and Intel optimized TF 2.x. Validated TensorFlow [Version](/docs/source/installation_guide.md#validated-software-environment).

# Prerequisite

## 1. Environment

### Installation
Recommend python 3.8 or higher version.

```shell
# Install Intel® Neural Compressor
pip install neural-compressor
```

### Install Tensorflow
```shell
pip install tensorflow
```

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

## 2. Prepare Dataset

  TensorFlow [models](https://github.com/tensorflow/models) repo provides [scripts and instructions](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data) to download. 
  This example uses the raw ImageNet data. Therefore, users do not need to convert the data to TF Record format.

# Run

## Run pretraining
```shell
bash run_distillation.sh --topology=mobilenet --teacher=densenet201 --dataset_location=/path/to/dataset --output_model=path/to/output_model
```

> Note: `--topology` is the student model and `--teacher` is the teacher model.

