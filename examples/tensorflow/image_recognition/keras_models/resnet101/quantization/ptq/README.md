Step-by-Step
============

This document is used to enable Tensorflow Keras models using Intel® Neural Compressor.


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
> Note: Supported Tensorflow [Version](../../../../../../../README.md).

### 3. Prepare Pretrained model

The pretrained model is provided by [Keras Applications](https://keras.io/api/applications/). prepare the model, Run as follow: 
 ```
 prepare_model.py   --output_model=/path/to/model
 ```
`--output_model ` the model should be saved as SavedModel format or H5 format.

## Run Command
  ```shell
  bash run_tuning.sh --config=resnet101.yaml --input_model=./path/to/model --output_model=./result --eval_data=/path/to/evaluation/dataset --calib_data=/path/to/calibration/dataset
  bash run_benchmark.sh --config=resnet101.yaml --input_model=./path/to/model --mode=performance --eval_data=/path/to/evaluation/dataset
  ```

