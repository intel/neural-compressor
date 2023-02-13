Step-by-Step
============

This document is used to enable Tensorflow Keras models using Intel® Neural Compressor.
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
> Note: Validated TensorFlow [Version](/docs/source/installation_guide.md#validated-software-environment).

### 3. Install Intel Extension for Tensorflow
#### Quantizing the model on Intel GPU
Intel Extension for Tensorflow is mandatory to be installed for quantizing the model on Intel GPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[gpu]
```
Please refer to the [Installation Guides](https://dgpu-docs.intel.com/installation-guides/ubuntu/ubuntu-focal-dc.html) for latest Intel GPU driver installation.
For any more details, please follow the procedure in [install-gpu-drivers](https://github.com/intel-innersource/frameworks.ai.infrastructure.intel-extension-for-tensorflow.intel-extension-for-tensorflow/blob/master/docs/install/install_for_gpu.md#install-gpu-drivers).

#### Quantizing the model on Intel CPU(Experimental)
Intel Extension for Tensorflow for Intel CPUs is experimental currently. It's not mandatory for quantizing the model on Intel CPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[cpu]
```

### 4. Prepare Pretrained model

The pretrained model is provided by [Keras Applications](https://keras.io/api/applications/). prepare the model, Run as follow: 
 ```
python prepare_model.py   --output_model=/path/to/model
 ```
`--output_model ` the model should be saved as SavedModel format or H5 format.

## Quantization Config
The Quantization Config class has default parameters setting for running on Intel CPUs. If running this example on Intel GPUs, the 'backend' parameter should be set to 'itex' and the 'device' parameter should be set to 'gpu'.

```
config = PostTrainingQuantConfig(
    device="gpu",
    backend="itex",
    ...
    )
```

## Run Command
  ```shell
  bash run_tuning.sh --input_model=./path/to/model --output_model=./result --dataset_location=/path/to/evaluation/dataset --batch_size=32
  bash run_benchmark.sh --input_model=./path/to/model --mode=performance --dataset_location=/path/to/evaluation/dataset --batch_size=1
  ```

