Step-by-Step
============

This document is used to enable Tensorflow efficientnet_v2_b0 SavedModel format using Intel® Neural Compressor.
This example can run on Intel CPUs and GPUs.


# Prerequisite

## 1. Environment

### Install Intel® Neural Compressor
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
Download the mobilenetv1 model from tensorflow-hub.

image recognition
- [efficientnet_v2_b0](https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/classification/2)

## 3. Prepare Dataset

Download [ImageNet](http://www.image-net.org/) Raw image to dir: /path/to/ImageNet. The dir include below folder and files:

```bash
ls /path/to/ImageNet
ILSVRC2012_img_val  val.txt
```

# Run Command
## 1. Quantization
  ```shell
  bash run_tuning.sh --input_model=./SavedModel --output_model=./nc_SavedModel --dataset_location=/path/to/ImageNet/
  ```

## 2. Benchmark
  ```shell
  bash run_benchmark.sh --input_model=./SavedModel --mode=accuracy --dataset_location=/path/to/ImageNet/ --batch_size=32
  bash run_benchmark.sh --input_model=./SavedModel --mode=performance --dataset_location=/path/to/ImageNet/ --batch_size=1
  ```
