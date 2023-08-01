Step-by-Step
============

This document list steps of reproducing mobilenet_v1 model tuning and benchmark results via Neural Compressor.
This example can run on Intel CPUs and GPUs.

> **Note**: 
> Most of those models are both supported in Intel optimized TF 1.15.x and Intel optimized TF 2.x. Validated TensorFlow [Version](/docs/source/installation_guide.md#validated-software-environment).
# Prerequisite

## 1. Environment

### Installation
Recommend python 3.6 or higher version.

```shell
# Install Intel® Neural Compressor
pip install neural-compressor
```

### Install Intel Tensorflow
```shell
pip install intel-tensorflow
```

### Installation Dependency packages
```shell
cd examples/tensorflow/object_detection/tensorflow_models/quantization/ptq
pip install -r requirements.txt
```

### Install Intel Extension for Tensorflow
#### Quantizing the model on Intel GPU(Mandatory to install ITEX)
Intel Extension for Tensorflow is mandatory to be installed for quantizing the model on Intel GPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[gpu]
```
For any more details, please follow the procedure in [install-gpu-drivers](https://github.com/intel/intel-extension-for-tensorflow/blob/main/docs/install/install_for_gpu.md#install-gpu-drivers)

#### Quantizing the model on Intel CPU(Optional to install ITEX)
Intel Extension for Tensorflow for Intel CPUs is experimental currently. It's not mandatory for quantizing the model on Intel CPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[cpu]
```
> **Note**: 
> The version compatibility of stock Tensorflow and ITEX can be checked [here](https://github.com/intel/intel-extension-for-tensorflow#compatibility-table). Please make sure you have installed compatible Tensorflow and ITEX.

## 2. Prepare pre-trained model

  Download pre-trained PB
  ```shell
  wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/mobilenet_v1_1.0_224_frozen.pb
  ```

## 3. Prepare Dataset

  TensorFlow [models](https://github.com/tensorflow/models) repo provides [scripts and instructions](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data) to download, process and convert the ImageNet dataset to the TF records format.
  We also prepared related scripts in ` examples/tensorflow/image_recognition/tensorflow_models/imagenet_prepare` directory. To download the raw images, the user must create an account with image-net.org. If you have downloaded the raw data and preprocessed the validation data by moving the images into the appropriate sub-directory based on the label (synset) of the image. we can use below command ro convert it to tf records format.

  ```shell
  cd examples/tensorflow/image_recognition/tensorflow_models/
  # convert validation subset
  bash prepare_dataset.sh --output_dir=./mobilenet_v1/quantization/ptq/data --raw_dir=/PATH/TO/img_raw/val/ --subset=validation
  # convert train subset
  bash prepare_dataset.sh --output_dir=./mobilenet_v1/quantization/ptq/data --raw_dir=/PATH/TO/img_raw/train/ --subset=train
  ```

# Run

## Quantization Config

The Quantization Config class has default parameters setting for running on Intel CPUs. If running this example on Intel GPUs, the 'backend' parameter should be set to 'itex' and the 'device' parameter should be set to 'gpu'.

```
config = PostTrainingQuantConfig(
    device="gpu",
    backend="itex",
    ...
    )
```

## 1 Quantization

  ```shell
  cd examples/tensorflow/image_recognition/tensorflow_models/mobilenet_v1/quantization/ptq
  bash run_quant.sh --input_model=/PATH/TO/mobilenet_v1_1.0_224_frozen.pb \
      --output_model=./nc_mobilenet_v1.pb --dataset_location=/path/to/ImageNet/
  ```

## 2. Benchmark
  ```shell
  cd examples/tensorflow/image_recognition/tensorflow_models/mobilenet_v1/quantization/ptq
  bash run_benchmark.sh --input_model=./nc_mobilenet_v1.pb --mode=accuracy --dataset_location=/path/to/ImageNet/ --batch_size=32
  bash run_benchmark.sh --input_model=./nc_mobilenet_v1.pb --mode=performance --dataset_location=/path/to/ImageNet/ --batch_size=1
  ```
