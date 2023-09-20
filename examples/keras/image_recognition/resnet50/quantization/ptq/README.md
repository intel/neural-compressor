Step-by-Step
============

This document is used to enable Tensorflow Keras Resnet50 model quantization and benchmark using Intel® Neural Compressor.
This example can run on Intel CPUs and GPUs.


# Prerequisite

## 1. Environment

### Installation
```shell
# Install Intel® Neural Compressor
pip install neural-compressor
```

### Install Requirements
The Tensorflow and intel-extension-for-tensorflow is mandatory to be installed to run this example.
The Intel Extension for Tensorflow for Intel CPUs is installed as default.
```shell
pip install -r requirements.txt
```
> Note: Validated TensorFlow [Version](/docs/source/installation_guide.md#validated-software-environment).

## 2. Prepare Pretrained model

The pretrained model is provided by [Keras Applications](https://keras.io/api/applications/). prepare the model, Run as follow: 
 ```
python prepare_model.py --output_model=/path/to/model
 ```
`--output_model ` the model should be saved as SavedModel format or H5 format.

## 3. Prepare Dataset

  TensorFlow [models](https://github.com/tensorflow/models) repo provides [scripts and instructions](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data) to download, process and convert the ImageNet dataset to the TF records format.
  We also prepared related scripts in `imagenet_prepare` directory. To download the raw images, the user must create an account with image-net.org. If you have downloaded the raw data and preprocessed the validation data by moving the images into the appropriate sub-directory based on the label (synset) of the image. we can use below command ro convert it to tf records format.

  ```shell
  cd examples/keras/image_recognition/
  # convert validation subset
  bash prepare_dataset.sh --output_dir=./resnet50/quantization/ptq/data --raw_dir=/PATH/TO/img_raw/val/ --subset=validation
  # convert train subset
  bash prepare_dataset.sh --output_dir=./resnet50/quantization/ptq/data --raw_dir=/PATH/TO/img_raw/train/ --subset=train
  cd resnet50/quantization/ptq
  ```
> **Note**: 
> The raw ImageNet dataset resides in JPEG files should be in the following directory structure. Taking validation set as an example:<br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;/PATH/TO/img_raw/val/n01440764/ILSVRC2012_val_00000293.JPEG<br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;/PATH/TO/img_raw/val/n01440764/ILSVRC2012_val_00000543.JPEG<br>
> where 'n01440764' is the unique synset label associated with these images.

# Run Command
## 1 Tune
  ```shell
  bash run_quant.sh --input_model=./resnet50_keras/ --output_model=./result --dataset_location=/path/to/evaluation/dataset
  ```

## Benchmark
  ```shell
  bash run_benchmark.sh --input_model=./result --dataset_location=/path/to/evaluation/dataset --mode=performance --batch_size=1
  bash run_benchmark.sh --input_model=./result --dataset_location=/path/to/evaluation/dataset --mode=accuracy --batch_size=32
  ```
