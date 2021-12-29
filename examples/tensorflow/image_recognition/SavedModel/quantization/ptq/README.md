Step-by-Step
============

This document is used to enable Tensorflow SavedModel format using Intel® Neural Compressor for performance only.


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
> Note: Supported Tensorflow >= 2.4.0.

### 3. Prepare Pretrained model
Download the model from tensorflow-hub.

image recognition
- [mobilenetv1](https://hub.tensorflow.google.cn/google/imagenet/mobilenet_v1_075_224/classification/5)
- [mobilenetv2](https://hub.tensorflow.google.cn/google/imagenet/mobilenet_v2_035_224/classification/5)
- [efficientnet_v2_b0](https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet1k_b0/classification/2)


## Run Command
  ```shell
  bash run_tuning.sh --config=./config.yaml --input_model=./SavedModel --output_model=./nc_SavedModel
  ```
  ```shell
  bash run_benchmark.sh --config=./config.yaml --input_model=./SavedModel --mode=performance
  ```