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

Download CoCo Dataset from [Official Website](https://cocodataset.org/#download).
The dataset can be converted into tensorflow records using the `https://github.com/tensorflow/models.git` dedicated script.

# Run Command

## Quantize Tensorflow FP32 model to Tensorflow INT8 QDQ model
```shell
bash run_tuning.sh --input_model=./faster_rcnn_resnet50_fp32_coco/frozen_inference_graph.pb --output_model=./faster_rcnn_resnet50_coco_int8.pb --dataset_location=/path/to/coco_dataset/
```

## Run benchmark for Tensorflow INT8 QDQ model
```shell
bash run_benchmark.sh --input_model=./faster_rcnn_resnet50_coco_int8.pb --mode=accuracy --dataset_location=/path/to/coco_dataset/ --batch_size=16
bash run_benchmark.sh --input_model=./faster_rcnn_resnet50_coco_int8.pb --mode=performance --dataset_location=/path/to/coco_dataset/ --batch_size=16
```
Please note this dataset is TF records format.

## Export Tensorflow INT8 QDQ model to ONNX INT8 QDQ model
```shell
bash run_export.sh --input_model=./faster_rcnn_resnet50_coco_int8.pb --output_model=./faster_rcnn_resnet50_coco_int8.onnx
```

## Run benchmark for ONNX INT8 QDQ model
```shell
bash run_benchmark.sh --input_model=./faster_rcnn_resnet50_coco_int8.onnx --mode=accuracy --dataset_location=/path/to/coco_dataset_raw/ --batch_size=16
bash run_benchmark.sh --input_model=./faster_rcnn_resnet50_coco_int8.onnx --mode=performance --dataset_location=/path/to/coco_dataset_raw/ --batch_size=16
```
Please note this dataset is Raw Coco dataset.
