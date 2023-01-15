Step-by-Step
============

This document is used to show how to export Tensorflow RestNet50 FP32 model to ONNX FP32 model using Intel® Neural Compressor.


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
wget https://zenodo.org/record/2535873/files/resnet50_v1.pb
```

### 4. Prepare Dataset

Download [ImageNet](http://www.image-net.org/) Raw image to dir: /path/to/ImageNet. The dir include below folder and files:

```bash
ls /path/to/ImageNet
ILSVRC2012_img_val  val.txt
```

## Run Command

### Export Tensorflow FP32 model to ONNX FP32 model
```shell
bash run_export.sh --input_model=./resnet50_v1.pb --output_model=./resnet50_v1.onnx
```

### Run benchmark for ONNX FP32 model
```shell
bash run_benchmark.sh --input_model=./resnet50_v1.onnx --mode=accuracy --dataset_location=/path/to/ImageNet/ --batch_size=32
bash run_benchmark.sh --input_model=./resnet50_v1.onnx --mode=performance --dataset_location=/path/to/ImageNet/ --batch_size=1
```
