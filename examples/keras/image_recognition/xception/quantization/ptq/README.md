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

### Install Requirements
The Tensorflow and intel-extension-for-tensorflow is mandatory to be installed to run this QAT example.
The Intel Extension for Tensorflow for Intel CPUs is installed as default.
```shell
pip install -r requirements.txt
```
> Note: Validated TensorFlow [Version](/docs/source/installation_guide.md#validated-software-environment).

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
  bash run_quant.sh --input_model=./path/to/model --output_model=./result --dataset_location=/path/to/evaluation/dataset --batch_size=32
  bash run_benchmark.sh --input_model=./path/to/model --mode=performance --dataset_location=/path/to/evaluation/dataset --batch_size=1
  ```

