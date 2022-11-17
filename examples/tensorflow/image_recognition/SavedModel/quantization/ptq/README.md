Step-by-Step
============

This document is used to enable Tensorflow SavedModel format using Intel® Neural Compressor for performance only.
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
> Note: Supported Tensorflow >= 2.4.0.

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
Download the model from tensorflow-hub.

image recognition
- [mobilenetv1](https://hub.tensorflow.google.cn/google/imagenet/mobilenet_v1_075_224/classification/5)
- [mobilenetv2](https://hub.tensorflow.google.cn/google/imagenet/mobilenet_v2_035_224/classification/5)
- [efficientnet_v2_b0](https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet1k_b0/classification/2)

## Write Yaml config file
In examples directory, there are mobilenet_v1.yaml, mobilenet_v2.yaml and efficientnet_v2_b0.yaml for tuning the model on Intel CPUs. The 'framework' in the yaml is set to 'tensorflow'. If running this example on Intel GPUs, the 'framework' should be set to 'tensorflow_itex' and the device in yaml file should be set to 'gpu'. The mobilenet_v1_itex.yaml, mobilenet_v2_itex.yaml and efficientnet_v2_b0_itex.yaml are prepared for the GPU case. We could remove most of items and only keep mandatory item for tuning. We also implement a calibration dataloader and have evaluation field for creation of evaluation function at internal neural_compressor.

## Run Command
  ```shell
  bash run_tuning.sh --config=./config.yaml --input_model=./SavedModel --output_model=./nc_SavedModel
  ```
  ```shell
  bash run_benchmark.sh --config=./config.yaml --input_model=./SavedModel --mode=performance
  ```