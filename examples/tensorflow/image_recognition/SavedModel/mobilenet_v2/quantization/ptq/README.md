Step-by-Step
============

This document is used to enable Tensorflow mobilenetv2 SavedModel format using Intel® Neural Compressor.
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
> Note: Supported Tensorflow >= 2.4.0.

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
Download the model from tensorflow-hub.

image recognition
- [mobilenetv2]((https://tfhub.dev/google/imagenet/mobilenet_v2_035_224/classification/5)

## 3. Prepare Dataset
TensorFlow [models](https://github.com/tensorflow/models) repo provides [scripts and instructions](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data) to download, process and convert the ImageNet dataset to the TF records format.
We also prepared related scripts in [TF image_recognition example](../../tensorflow/image_recognition/tensorflow_models/quantization/ptq/README.md#2-prepare-dataset).

# Run Command
## 1. Quantization
  ```shell
  bash run_tuning.sh --input_model=./SavedModel --output_model=./nc_SavedModel --dataset_location=/path/to/imagenet/
  ```

## 2. Benchmark
  ```shell
  bash run_benchmark.sh --input_model=./SavedModel --mode=accuracy --dataset_location=/path/to/imagenet/ --batch_size=32
  bash run_benchmark.sh --input_model=./SavedModel --mode=performance --dataset_location=/path/to/imagenet/ --batch_size=1
  ```
