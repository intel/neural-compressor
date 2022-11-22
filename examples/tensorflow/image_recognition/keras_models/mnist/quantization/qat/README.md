Step-by-Step
============

This document is used to list steps of reproducing TensorFlow keras Intel® Neural Compressor QAT conversion.
This example can run on Intel CPUs and GPUs.


## Prerequisite

### 1. Installation
```shell
# Install Intel® Neural Compressor
pip install neural-compressor
```
### 2. Install Intel Tensorflow and TensorFlow Model Optimization
```shell
pip install intel-tensorflow==2.4.0
pip install tensorflow_model_optimization==0.5.0
```
> Note: To generate correct qat model with tensorflow_model_optimization 0.5.0, pls use TensorFlow 2.4 or above.

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

Run the `train.py` script to get pretrained fp32 model.

### 5. Prepare QAT model

Run the `qat.py` script to get QAT model which in fact is a fp32 model with quant/dequant pair inserted.

## Write Yaml config file
In examples directory, there is a mnist.yaml for tuning the model on Intel CPUs. The 'framework' in the yaml is set to 'tensorflow'. If running this example on Intel GPUs, the 'framework' should be set to 'tensorflow_itex' and the device in yaml file should be set to 'gpu'. The mnist_itex.yaml is prepared for the GPU case. We could remove most of items and only keep mandatory item for tuning. We also implement a calibration dataloader and have evaluation field for creation of evaluation function at internal neural_compressor.

## Run Command
  ```shell
  python convert.py     # to convert QAT model to quantized model.

  python benchmark.py   # to run accuracy benchmark.
  ```

