Step-by-Step
============

This document is used to show how to export Tensorflow INT8 QDQ model to ONNX INT8 QDQ model using Intel® Neural Compressor.


## Prerequisite

### 1. Installation
```shell
# Install Intel® Neural Compressor
pip install neural-compressor
```
### 2. Install requirements
```shell
pip install -r requirements.txt
```

### 3. Prepare Pretrained model

```bash
export MODEL=ssd_mobilenet_v1_coco_2018_01_28
wget http://download.tensorflow.org/models/object_detection/$MODEL.tar.gz
tar -xvf $MODEL.tar.gz
```

### 4. Prepare Dataset

Download CoCo Dataset from [Official Website](https://cocodataset.org/#download).

## Run Command

### Quantize Tensorflow FP32 model to Tensorflow INT8 QDQ model
```shell
bash run_tuning.sh --input_model=./ssd_mobilenet_v1_coco_2018_01_28 --output_model=./ssd_mobilenet_v1_coco_2018_01_28_int8.pb --dataset_location=/path/to/coco_dataset/
```

### Run benchmark for Tensorflow INT8 QDQ model
```shell
bash run_benchmark.sh --input_model=./ssd_mobilenet_v1_coco_2018_01_28_int8.pb --mode=accuracy --dataset_location=/path/to/coco_dataset/ --batch_size=16
bash run_benchmark.sh --input_model=./ssd_mobilenet_v1_coco_2018_01_28_int8.pb --mode=performance --dataset_location=/path/to/coco_dataset/ --batch_size=16
```
Please note this dataset is TF records format.

### Export Tensorflow INT8 QDQ model to ONNX INT8 QDQ model
```shell
bash run_export.sh --input_model=./ssd_mobilenet_v1_coco_2018_01_28_int8.pb --output_model=./ssd_mobilenet_v1_coco_2018_01_28_int8.onnx
```

### Run benchmark for ONNX INT8 QDQ model
```shell
bash run_benchmark.sh --input_model=./ssd_mobilenet_v1_coco_2018_01_28_int8.onnx --mode=accuracy --dataset_location=/path/to/coco_dataset_raw/ --batch_size=16
bash run_benchmark.sh --input_model=./ssd_mobilenet_v1_coco_2018_01_28_int8.onnx --mode=performance --dataset_location=/path/to/coco_dataset_raw/ --batch_size=16
```
Please note this dataset is Raw Coco dataset.
