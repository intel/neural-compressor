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
#### Running the model on Intel CPU(Optional to install ITEX)
Intel Extension for Tensorflow for Intel CPUs is experimental currently. It's not mandatory for running the model on Intel CPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[cpu]
```
> **Note**: 
> The version compatibility of stock Tensorflow and ITEX can be checked [here](https://github.com/intel/intel-extension-for-tensorflow#compatibility-table). Please make sure you have installed compatible Tensorflow and ITEX.

## 2. Prepare Dataset

  TensorFlow [models](https://github.com/tensorflow/models) repo provides [scripts and instructions](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data) to download. 
  This example uses the raw ImageNet data. Therefore, users do not need to convert the data to TF Record format.

  The data folder is expected to contain subfolders representing the classes to which
    its images belong.

    Please arrange data in this way:
        dataset/class_1/xxx.png
        dataset/class_1/xxy.png
        dataset/class_1/xxz.png
        ...
        dataset/class_n/123.png
        dataset/class_n/nsdf3.png
        dataset/class_n/asd932_.png
    Please put images of different categories into different folders.

# Run

## Run pretraining
```shell
bash run_distillation.sh --topology=mobilenet --teacher=densenet201 --dataset_location=/path/to/dataset --output_model=path/to/output_model
```

> Note: `--topology` is the student model and `--teacher` is the teacher model.

