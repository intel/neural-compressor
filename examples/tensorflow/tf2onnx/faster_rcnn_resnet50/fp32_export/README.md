Step-by-Step
============

This document is used to show how to export Tensorflow faster_rcnn_resnet50 FP32 model to ONNX FP32 model using Intel® Neural Compressor.


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

## Export Tensorflow FP32 model to ONNX FP32 model
```shell
bash run_export.sh --input_model=./faster_rcnn_resnet50_fp32_coco/frozen_inference_graph.pb --output_model=./faster_rcnn_resnet50_fp32_coco.onnx
```

## Run benchmark for Tensorflow FP32 model
```shell
bash run_benchmark.sh --input_model=./faster_rcnn_resnet50_fp32_coco/frozen_inference_graph.pb --mode=accuracy --dataset_location=/path/to/coco_dataset/ --batch_size=16
bash run_benchmark.sh --input_model=./faster_rcnn_resnet50_fp32_coco/frozen_inference_graph.pb --mode=performance --dataset_location=/path/to/coco_dataset/ --batch_size=16
```
Please note this dataset is TF records format.

## Run benchmark for ONNX FP32 model
```shell
bash run_benchmark.sh --input_model=./faster_rcnn_resnet50_fp32_coco.onnx --mode=accuracy --dataset_location=/path/to/coco_dataset/ --batch_size=16
bash run_benchmark.sh --input_model=./faster_rcnn_resnet50_fp32_coco.onnx --mode=performance --dataset_location=/path/to/coco_dataset/ --batch_size=16
```
Please note this dataset is Raw Coco dataset.
