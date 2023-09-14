Step-by-Step
============

This document is used to show how to export Tensorflow INT8 QDQ model to ONNX INT8 QDQ model using Intel® Neural Compressor.
> Note: Validated Framework [Versions](/docs/source/installation_guide.md#validated-software-environment).

# Prerequisite

## 1. Environment

### Installation
```shell
# Install Intel® Neural Compressor
pip install neural-compressor
```

### Install requirements
The Tensorflow and intel-extension-for-tensorflow is mandatory to be installed to run this export ONNX INT8 model example.
The Intel Extension for Tensorflow for Intel CPUs is installed as default.
```shell
pip install -r requirements.txt
```

### Install Intel Extension for Tensorflow
Intel Extension for Tensorflow is mandatory to be installed for exporting Tensorflow model to ONNX.
```shell
pip install --upgrade intel-extension-for-tensorflow[cpu]
```

## 2 Prepare Pretrained model

```shell
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/resnet50_fp32_pretrained_model.pb
```


## 3. Prepare Dataset

  TensorFlow [models](https://github.com/tensorflow/models) repo provides [scripts and instructions](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data) to download, process and convert the ImageNet dataset to the TF records format.
  We also prepared related scripts in `imagenet_prepare` directory. To download the raw images, the user must create an account with image-net.org. If you have downloaded the raw data and preprocessed the validation data by moving the images into the appropriate sub-directory based on the label (synset) of the image. we can use below command ro convert it to tf records format.

  ```shell
  cd examples/tensorflow/image_recognition/tensorflow_models/
  # convert validation subset
  bash prepare_imagenet_dataset.sh --output_dir=/path/to/imagenet/ --raw_dir=/PATH/TO/img_raw/val/ --subset=validation
  # convert train subset
  bash prepare_imagenet_dataset.sh --output_dir=/path/to/imagenet/ --raw_dir=/PATH/TO/img_raw/train/ --subset=train
  cd resnet50_v1.0/export
  ```
> **Note**: 
> The raw ImageNet dataset resides in JPEG files should be in the following directory structure. Taking validation set as an example:<br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;/PATH/TO/img_raw/val/n01440764/ILSVRC2012_val_00000293.JPEG<br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;/PATH/TO/img_raw/val/n01440764/ILSVRC2012_val_00000543.JPEG<br>
> where 'n01440764' is the unique synset label associated with these images.

# Run Command
Please note the dataset is TF records format for running quantization and benchmark.

### Export Tensorflow FP32 model to ONNX FP32 model
```shell
bash run_export.sh --input_model=./resnet50_fp32_pretrained_model.pb --output_model=./resnet50_v1.onnx --dtype=fp32 --quant_format=qdq
```

## Run benchmark for Tensorflow FP32 model
```shell
bash run_benchmark.sh --input_model=./resnet50_fp32_pretrained_model.pb --mode=accuracy --dataset_location=/path/to/imagenet/ --batch_size=32
bash run_benchmark.sh --input_model=./resnet50_fp32_pretrained_model.pb --mode=performance --dataset_location=/path/to/imagenet/ --batch_size=1
```

### Run benchmark for ONNX FP32 model
```shell
bash run_benchmark.sh --input_model=./resnet50_v1.onnx --mode=accuracy --dataset_location=/path/to/imagenet/ --batch_size=32
bash run_benchmark.sh --input_model=./resnet50_v1.onnx --mode=performance --dataset_location=/path/to/imagenet/ --batch_size=1
```

### Export Tensorflow INT8 QDQ model to ONNX INT8 QDQ model
```shell
bash run_export.sh --input_model=./resnet50_fp32_pretrained_model.pb --output_model=./resnet50_v1_int8.onnx --dtype=int8 --quant_format=qdq --dataset_location=/path/to/imagenet/
```

## Run benchmark for Tensorflow INT8 model
```shell
bash run_benchmark.sh --input_model=./tf-quant.pb --mode=accuracy --dataset_location=/path/to/imagenet/ --batch_size=32
bash run_benchmark.sh --input_model=./tf-quant.pb --mode=performance --dataset_location=/path/to/imagenet/ --batch_size=1
```

### Run benchmark for ONNX INT8 QDQ model
```shell
bash run_benchmark.sh --input_model=./resnet50_v1_int8.onnx --mode=accuracy --dataset_location=/path/to/imagenet/ --batch_size=32
bash run_benchmark.sh --input_model=./resnet50_v1_int8.onnx --mode=performance --dataset_location=/path/to/imagenet/ --batch_size=1
