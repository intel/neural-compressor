Step-by-Step
============

This document is used to list steps of reproducing Intel® Neural Compressor QAT feature.


## Prerequisite

### 1. Installation
```shell
# Install Intel® Neural Compressor
pip install neural-compressor
```
### 2. Install Tensorflow.
```shell
pip install tensorflow
```

## Run Command
```shell
python resnet_v2.py    # to get the quantized ResNet-V2 model which will be saved into './trained_qat_model'.
```

