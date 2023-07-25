Step-by-Step
============

This document is used to enable Tensorflow efficientnet_v2_b0 SavedModel format using Intel® Neural Compressor.
This example can run on Intel CPUs and GPUs.

> **Note**: 
> Most of those models are both supported in Intel optimized TF 1.15.x and Intel optimized TF 2.x. Validated TensorFlow [Version](/docs/source/installation_guide.md#validated-software-environment).

# Prerequisite

## 1. Environment

### Install Intel® Neural Compressor
```shell
pip install neural-compressor
```
### Install Tensorflow
```shell
pip install tensorflow
```

### Install Intel Extension for Tensorflow
#### Quantizing the model on Intel GPU(Mandatory to install ITEX)
Intel Extension for Tensorflow is mandatory to be installed for quantizing the model on Intel GPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[gpu]
```
For any more details, please follow the procedure in [install-gpu-drivers](https://github.com/intel/intel-extension-for-tensorflow/blob/main/docs/install/install_for_gpu.md#install-gpu-drivers)

#### Quantizing the model on Intel CPU(Optional to install ITEX)
Intel Extension for Tensorflow for Intel CPUs is experimental currently. It's not mandatory for quantizing the model on Intel CPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[cpu]
```
> **Note**: 
> The version compatibility of stock Tensorflow and ITEX can be checked [here](https://github.com/intel/intel-extension-for-tensorflow#compatibility-table). Please make sure you have installed compatible Tensorflow and ITEX.

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

## Quantization Config

The Quantization Config class has default parameters setting for running on Intel CPUs. If running this example on Intel GPUs, the 'backend' parameter should be set to 'itex' and the 'device' parameter should be set to 'gpu'.

```
config = PostTrainingQuantConfig(
    device="gpu",
    backend="itex",
    ...
    )
```

## 1. Quantization
  ```shell
  bash run_quant.sh --input_model=./SavedModel --output_model=./nc_SavedModel --dataset_location=/path/to/ImageNet/
  ```

## 2. Benchmark
  ```shell
  bash run_benchmark.sh --input_model=./SavedModel --mode=accuracy --dataset_location=/path/to/ImageNet/ --batch_size=32
  bash run_benchmark.sh --input_model=./SavedModel --mode=performance --dataset_location=/path/to/ImageNet/ --batch_size=1
  ```
