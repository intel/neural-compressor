Step-by-Step
============

This document is used to enable Tensorflow Keras resnetv2_101 model quantization and benchmark using Intel® Neural Compressor.
This example can run on Intel CPUs and GPUs.


# Prerequisite

## 1. Environment

### Installation
```shell
# Install Intel® Neural Compressor
pip install neural-compressor
```

### Install Requirements
The Tensorflow and intel-extension-for-tensorflow is mandatory to be installed to run this QAT example.
The Intel Extension for Tensorflow for Intel CPUs is installed as default.
```shell
pip install -r requirements.txt
```
> Note: Validated TensorFlow [Version](/docs/source/installation_guide.md#validated-software-environment).

## 2. Prepare Pretrained model

The pretrained model is provided by [Keras Applications](https://keras.io/api/applications/). prepare the model, Run as follow: 
 ```
python prepare_model.py   --output_model=/path/to/model
 ```
`--output_model ` the model should be saved as SavedModel format or H5 format.

## 3. Prepare Dataset

  TensorFlow [models](https://github.com/tensorflow/models) repo provides [scripts and instructions](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data) to download, process and convert the ImageNet dataset to the TF records format.
  We also prepared related scripts in `imagenet_prepare` directory. To download the raw images, the user must create an account with image-net.org. If you have downloaded the raw data and preprocessed the validation data by moving the images into the appropriate sub-directory based on the label (synset) of the image. we can use below command ro convert it to tf records format.

  ```shell
  cd examples/tensorflow/image_recognition/keras_models/
  # convert validation subset
  bash prepare_dataset.sh --output_dir=/resnetv2_101/quantization/ptq/data --raw_dir=/PATH/TO/img_raw/val/ --subset=validation
  # convert train subset
  bash prepare_dataset.sh --output_dir=/resnetv2_101/quantization/ptq/data --raw_dir=/PATH/TO/img_raw/train/ --subset=train
  cd resnetv2_101/quantization/ptq
  ```
> **Note**: 
> The raw ImageNet data set resides in JPEG files should located in the following directory structure.<br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;data_dir/n01440764/ILSVRC2012_val_00000293.JPEG<br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;data_dir/n01440764/ILSVRC2012_val_00000543.JPEG<br>
> where 'n01440764' is the unique synset label associated with these images.

# Run Command

## Quantization Config
The Quantization Config class has default parameters setting for running on Intel CPUs. If running this example on Intel GPUs, the 'backend' parameter should be set to 'itex' and the 'device' parameter should be set to 'gpu'.

```
config = PostTrainingQuantConfig(
    device="gpu",
    backend="itex",
    ...
    )
```

## Quantization
  ```shell
  bash run_quant.sh --input_model=./vgg16_keras/ --output_model=./result --dataset_location=/path/to/evaluation/dataset
  ```

## Benchmark
  ```shell
  bash run_benchmark.sh --input_model=./result --mode=accuracy --dataset_location=/path/to/evaluation/dataset --batch_size=32
  bash run_benchmark.sh --input_model=./result --mode=performance --dataset_location=/path/to/evaluation/dataset --batch_size=1
  ```
