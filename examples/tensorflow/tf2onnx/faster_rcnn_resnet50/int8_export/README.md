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

```shell
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/faster_rcnn_resnet50_fp32_coco_pretrained_model.tar.gz
tar -xvf faster_rcnn_resnet50_fp32_coco_pretrained_model.tar.gz
```

## 3. Prepare Dataset

### Automatic dataset download

> **_Note: `prepare_coco_dataset.sh` script works with TF version 1.x._**

Run the `prepare_coco_dataset.sh` script located in `examples/tensorflow/tf2onnx`.

Usage:
```shell
cd examples/tensorflow/tf2onnx/
bash prepare_coco_dataset.sh
cd faster_rcnn_resnet50/int8_export
```

This script will download the *train*, *validation* and *test* COCO datasets. Furthermore it will convert them to
tensorflow records using the `https://github.com/tensorflow/models.git` dedicated script.

### Manual dataset download
Download CoCo Dataset from [Official Website](https://cocodataset.org/#download).

# Run Command
Please note the dataset is TF records format for running quantization and benchmark.

## Quantize Tensorflow FP32 model to Tensorflow INT8 QDQ model
```shell
bash run_tuning.sh --input_model=./faster_rcnn_resnet50_fp32_coco/frozen_inference_graph.pb --output_model=./faster_rcnn_resnet50_coco_int8.pb --dataset_location=/path/to/coco_dataset/
```

## Run benchmark for Tensorflow INT8 QDQ model
```shell
bash run_benchmark.sh --input_model=./faster_rcnn_resnet50_coco_int8.pb --mode=accuracy --dataset_location=/path/to/coco_dataset/ --batch_size=16
bash run_benchmark.sh --input_model=./faster_rcnn_resnet50_coco_int8.pb --mode=performance --dataset_location=/path/to/coco_dataset/ --batch_size=16
```

## Export Tensorflow INT8 QDQ model to ONNX INT8 QDQ model
```shell
bash run_export.sh --input_model=./faster_rcnn_resnet50_coco_int8.pb --output_model=./faster_rcnn_resnet50_coco_int8.onnx
```

## Run benchmark for ONNX INT8 QDQ model
```shell
bash run_benchmark.sh --input_model=./faster_rcnn_resnet50_coco_int8.onnx --mode=accuracy --dataset_location=/path/to/coco_dataset/ --batch_size=16
bash run_benchmark.sh --input_model=./faster_rcnn_resnet50_coco_int8.onnx --mode=performance --dataset_location=/path/to/coco_dataset/ --batch_size=16
```
