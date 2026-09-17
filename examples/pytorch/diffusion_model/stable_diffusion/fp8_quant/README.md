Step-by-Step
============

This document describes the step-by-step instructions for FP8 quantization for [stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1) with Intel® Neural Compressor.


# Prerequisite

### 1. Environment

```shell
bash setup.sh
pip install -r requirements.txt
```

### 2. Prepare Dataset

```bash
DATASET_DIR=${YOUR_DATA_PATH}

dir=$(pwd)
mkdir ${DATASET_DIR}; cd ${DATASET_DIR}
curl -O http://images.cocodataset.org/zips/val2017.zip; unzip val2017.zip
curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip; unzip annotations_trainval2017.zip
cd $dir
```


### 3. Prepare pretrained model

huggingface-cli download --local-dir ${YOUR_PATH}/stable-diffusion stabilityai/stable-diffusion-2-1

# Run with CPU

```shell
TORCHINDUCTOR_FREEZING=1 python main.py --model_path /path/to/stable-diffusion --data_path /path/to/dataset --calib --quant --accuracy
```
or only do quantization after calibration is done
```shell
TORCHINDUCTOR_FREEZING=1 python main.py --model_path /path/to/stable-diffusion --data_path /path/to/dataset  --quant --accuracy
```
