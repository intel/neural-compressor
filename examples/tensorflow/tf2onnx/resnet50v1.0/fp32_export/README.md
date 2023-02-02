Step-by-Step
============

This document is used to show how to export Tensorflow RestNet50_V1.0 FP32 model to ONNX FP32 model using Intel® Neural Compressor.


# Prerequisite

## 1. Environment

### Installation
```shell
# Install Intel® Neural Compressor
pip install neural-compressor
```

### Install requirements
```shell
pip install -r requirements.txt
```

### Install Intel Extension for Tensorflow
Intel Extension for Tensorflow is mandatory to be installed for exporting Tensorflow model to ONNX.
```shell
pip install --upgrade intel-extension-for-tensorflow[cpu]
```

### 2. Prepare Pretrained model

```shell
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/resnet50_fp32_pretrained_model.pb
```

### 3. Prepare Dataset

Download [ImageNet](http://www.image-net.org/) Raw image to dir: /path/to/ImageNet. The dir include below folder and files:

```bash
ls /path/to/ImageNet
ILSVRC2012_img_val  val.txt
```

# Run Command

## Export Tensorflow FP32 model to ONNX FP32 model
```shell
bash run_export.sh --input_model=./resnet50_fp32_pretrained_model.pb --output_model=./resnet50_v1.onnx
```

## Run benchmark for Tensorflow FP32 model
```shell
bash run_benchmark.sh --input_model=./resnet50_fp32_pretrained_model.pb --mode=accuracy --dataset_location=/path/to/imagenet/ --batch_size=32
bash run_benchmark.sh --input_model=./resnet50_fp32_pretrained_model.pb --mode=performance --dataset_location=/path/to/imagenet/ --batch_size=1
```

## Run benchmark for ONNX FP32 model
```shell
bash run_benchmark.sh --input_model=./resnet50_v1.onnx --mode=accuracy --dataset_location=/path/to/imagenet/ --batch_size=32
bash run_benchmark.sh --input_model=./resnet50_v1.onnx --mode=performance --dataset_location=/path/to/imagenet/ --batch_size=1
```
