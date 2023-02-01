Step-by-Step
============

This document is used to show how to export Tensorflow ssd_mobilenet_v1 FP32 model to ONNX FP32 model using Intel® Neural Compressor.


# Prerequisite

## 1. Environment

### Installation
Recommend python 3.8 or higher version.
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

## 2. Prepare Pretrained model

```shell
export MODEL=ssd_mobilenet_v1_coco_2018_01_28
wget http://download.tensorflow.org/models/object_detection/$MODEL.tar.gz
tar -xvf $MODEL.tar.gz
```

## 3. Prepare Dataset

Download CoCo Dataset from [Official Website](https://cocodataset.org/#download).
The dataset can be converted into tensorflow records using the `https://github.com/tensorflow/models.git` dedicated script.

# Run Command

## Export Tensorflow FP32 model to ONNX FP32 model
```shell
bash run_export.sh --input_model=./ssd_mobilenet_v1_coco_2018_01_28 --output_model=./ssd_mobilenet_v1_coco_2018_01_28.onnx
```

## Run benchmark for Tensorflow FP32 model
```shell
bash run_benchmark.sh --input_model=./ssd_mobilenet_v1_coco_2018_01_28 --mode=accuracy --dataset_location=/path/to/coco_dataset/ --batch_size=16
bash run_benchmark.sh --input_model=./ssd_mobilenet_v1_coco_2018_01_28 --mode=performance --dataset_location=/path/to/coco_dataset/ --batch_size=16
```
Please note this dataset is TF records format.

## Run benchmark for ONNX FP32 model
```shell
bash run_benchmark.sh --input_model=./ssd_mobilenet_v1_coco_2018_01_28.onnx --mode=accuracy --dataset_location=/path/to/coco_dataset/ --batch_size=16
bash run_benchmark.sh --input_model=./ssd_mobilenet_v1_coco_2018_01_28.onnx --mode=performance --dataset_location=/path/to/coco_dataset/ --batch_size=16
```
Please note this dataset is Raw Coco dataset.
