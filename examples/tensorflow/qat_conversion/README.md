Step-by-Step
============

This document is used to list steps of reproducing TensorFlow keras Intel® Neural Compressor QAT conversion.


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

### 3. Prepare Pretrained model

Run the `train.py` script to get pretrained fp32 model.

### 4. Prepare QAT model

Run the `qat.py` script to get QAT model which in fact is a fp32 model with quant/dequant pair inserted.

## Run Command
  ```shell
  python convert.py     # to convert QAT model to quantized model.

  python benchmark.py   # to run accuracy benchmark.
  ```

