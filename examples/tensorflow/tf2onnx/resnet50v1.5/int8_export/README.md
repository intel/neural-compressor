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
The Tensorflow and intel-extension-for-tensorflow is mandatory to be installed to run this export ONNX INT8 model example.
The Intel Extension for Tensorflow for Intel CPUs is installed as default.
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
The Raw image dataset is used for running benchmarking for ONNX model.

TensorFlow [models](https://github.com/tensorflow/models) repo provides [scripts and instructions](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data) to download, process and convert the ImageNet dataset to the TF records format. The TF records format dataset is used for quantizing Tensorflow FP32 model to Tensorflow INT8 QDQ model.

## Run Command

### Quantize Tensorflow FP32 model to Tensorflow INT8 QDQ model
```shell
bash run_tuning.sh --input_model=./resnet50_v1.pb --output_model=./resnet50_v1_int8.pb --dataset_location=/path/to/imagenet/
```
Please note this dataset is TF records format.

### Run benchmark for Tensorflow INT8 model
```shell
bash run_benchmark.sh --input_model=./resnet50_v1_int8.pb --mode=accuracy --dataset_location=/path/to/imagenet/ --batch_size=32
bash run_benchmark.sh --input_model=./resnet50_v1_int8.pb --mode=performance --dataset_location=/path/to/imagenet/ --batch_size=1
```
Please note this dataset is TF records format.

### Export Tensorflow INT8 QDQ model to ONNX INT8 QDQ model
```shell
bash run_export.sh --input_model=./resnet50_v1_int8.pb --output_model=./resnet50_v1_int8.onnx
```

### Run benchmark for ONNX INT8 QDQ model
```shell
bash run_benchmark.sh --input_model=./resnet50_v1_int8.onnx --mode=accuracy --dataset_location=/path/to/ImageNet/ --batch_size=32
bash run_benchmark.sh --input_model=./resnet50_v1_int8.onnx --mode=performance --dataset_location=/path/to/ImageNet/ --batch_size=1
```
Please note this dataset is Raw image dataset.
