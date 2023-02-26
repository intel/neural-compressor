Step-by-Step
============

This document is used to show how to export Tensorflow INT8 QDQ model to ONNX INT8 QDQ model using Intel® Neural Compressor.


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

```bash
export MODEL=ssd_mobilenet_v1_coco_2018_01_28
wget http://download.tensorflow.org/models/object_detection/$MODEL.tar.gz
tar -xvf $MODEL.tar.gz
```

## 3. Prepare Dataset

### Automatic dataset download

> **_Note: `prepare_dataset.sh` script works with TF version 1.x._**

Run the `prepare_dataset.sh` script located in `examples/tensorflow/object_detection/tensorflow_models`.

Usage:
```shell
cd ./examples/tensorflow/object_detection/tensorflow_models
bash prepare_dataset.sh
cd ssd_mobilenet_v1/export
```

This script will download the *train*, *validation* and *test* COCO datasets. Furthermore it will convert them to
tensorflow records using the `https://github.com/tensorflow/models.git` dedicated script.

### Manual dataset download
Download CoCo Dataset from [Official Website](https://cocodataset.org/#download).

# Run Command
Please note the dataset is TF records format for running quantization and benchmark.

### Export Tensorflow FP32 model to ONNX FP32 model
```shell
bash run_export.sh --input_model=./ssd_mobilenet_v1_coco_2018_01_28 --output_model=./ssd_mobilenet_v1_coco_2018_01_28.onnx --dtype=fp32 --quant_format=qdq
```

## Run benchmark for Tensorflow FP32 model
```shell
bash run_benchmark.sh --input_model=./ssd_mobilenet_v1_coco_2018_01_28 --mode=accuracy --dataset_location=/path/to/coco_dataset/ --batch_size=32
bash run_benchmark.sh --input_model=./ssd_mobilenet_v1_coco_2018_01_28 --mode=performance --dataset_location=/path/to/coco_dataset/ --batch_size=1
```

### Run benchmark for ONNX FP32 model
```shell
bash run_benchmark.sh --input_model=./ssd_mobilenet_v1_coco_2018_01_28.onnx --mode=accuracy --dataset_location=/path/to/coco_dataset/ --batch_size=32
bash run_benchmark.sh --input_model=./ssd_mobilenet_v1_coco_2018_01_28.onnx --mode=performance --dataset_location=/path/to/coco_dataset/ --batch_size=1
```

### Export Tensorflow INT8 QDQ model to ONNX INT8 QDQ model
```shell
bash run_export.sh --input_model=./ssd_mobilenet_v1_coco_2018_01_28 --output_model=./ssd_mobilenet_v1_coco_2018_01_28_int8.onnx --dtype=int8 --quant_format=qdq --dataset_location=/path/to/coco_dataset/
```

## Run benchmark for Tensorflow INT8 model
```shell
bash run_benchmark.sh --input_model=./tf-quant.pb --mode=accuracy --dataset_location=/path/to/coco_dataset/ --batch_size=32
bash run_benchmark.sh --input_model=./tf-quant.pb --mode=performance --dataset_location=/path/to/coco_dataset/ --batch_size=1
```

### Run benchmark for ONNX INT8 QDQ model
```shell
bash run_benchmark.sh --input_model=./ssd_mobilenet_v1_coco_2018_01_28_int8.onnx --mode=accuracy --dataset_location=/path/to/coco_dataset/ --batch_size=32
bash run_benchmark.sh --input_model=./ssd_mobilenet_v1_coco_2018_01_28_int8.onnx --mode=performance --dataset_location=/path/to/coco_dataset/ --batch_size=1
```