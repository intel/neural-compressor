Step-by-Step
============

This example load an object detection model converted from [ONNX Model Zoo](https://github.com/onnx/models) and confirm its accuracy and speed based on [cityscapes dataset](https://www.cityscapes-dataset.com/downloads/).

# Prerequisite

## 1. Environment
```shell
pip install neural-compressor
pip install -r requirements.txt
```
> Note: Validated ONNX Runtime [Version](/docs/source/installation_guide.md#validated-software-environment).

## 2. Prepare Model

Download model from [ONNX Model Zoo](https://github.com/onnx/models).

```shell
wget https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/duc/model/ResNet101-DUC-12.onnx
```

## 3. Prepare Dataset
Download dataset [cityscapes dataset](https://www.cityscapes-dataset.com/downloads/).

Dataset directories:

```bash
cityscapes
├── gtFine
|   └── val
├── leftImg8bit
|   └── val
```

# Run

## 1. Quantization

Static quantization with QOperator format:

```bash
bash run_quant.sh --input_model=path/to/model  \ # model path as *.onnx
                   --output_model=path/to/save \ # model path as *.onnx
                   --dataset_location=/path/to/cityscapes/leftImg8bit/val \
                   --quant_format="QOperator"
```

## 2. Benchmark

```bash
bash run_benchmark.sh --input_model=path/to/model \  # model path as *.onnx
                      --dataset_location=/path/to/cityscapes/leftImg8bit/val \
                      --mode=performance
```
