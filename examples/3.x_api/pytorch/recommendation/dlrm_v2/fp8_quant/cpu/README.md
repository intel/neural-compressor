Step-by-Step
============

This document describes the step-by-step instructions for FP8 quantization for [DLRM v2](https://github.com/facebookresearch/dlrm/tree/main/torchrec_dlrm) with Intel® Neural Compressor.


# Prerequisite

### 1. Environment

```shell
bash steup.sh
pip install -r requirements.txt
```

### 2. Prepare Dataset

You can download preprocessed dataset by following
https://github.com/mlcommons/inference/tree/master/recommendation/dlrm_v2/pytorch#download-preprocessed-dataset


### 3. Prepare pretrained model

You can download and unzip checkpoint by following
https://github.com/mlcommons/inference/tree/master/recommendation/dlrm_v2/pytorch#downloading-model-weights


# Run with CPU

```shell
TORCHINDUCTOR_FREEZING=1 python main.py --model_path /path/to/model_weights --data_path /path/to/dataset --calib --quant --accuracy
```
or only do quantization after calibration is done
```shell
TORCHINDUCTOR_FREEZING=1 python main.py --model_path /path/to/model_weights --data_path /path/to/dataset  --quant --accuracy
```

