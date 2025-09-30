Step-by-Step
============

This document is used to list steps of reproducing TensorFlow Intel® Neural Compressor tuning zoo result of 3dunet-mlperf.
This example can run on Intel CPUs and GPUs.

# Prerequisite

## 1. Environment

### Installation
```shell
# Install Intel® Neural Compressor
pip install neural-compressor
```

### Install requirements
```shell
pip install -r requirements.txt
```
> Note: Validated TensorFlow [Version](/docs/source/installation_guide.md#validated-software-environment).

### Install Intel Extension for Tensorflow
#### Quantizing the model on Intel GPU(Mandatory to install ITEX)
Intel Extension for Tensorflow is mandatory to be installed for quantizing the model on Intel GPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[xpu]
```
Please refer to the [Installation Guides](https://dgpu-docs.intel.com/installation-guides/ubuntu/ubuntu-focal-dc.html) for latest Intel GPU driver installation.
For any more details, please follow the procedure in [install-gpu-drivers](https://github.com/intel/intel-extension-for-tensorflow/blob/main/docs/install/install_for_xpu.md#install-gpu-drivers).

#### Quantizing the model on Intel CPU(Optional to install ITEX)
Intel Extension for Tensorflow for Intel CPUs is experimental currently. It's not mandatory for quantizing the model on Intel CPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[cpu]
```

> **Note**: 
> The version compatibility of stock Tensorflow and ITEX can be checked [here](https://github.com/intel/intel-extension-for-tensorflow#compatibility-table). Please make sure you have installed compatible Tensorflow and ITEX.

## 2. Prepare Pre-trained model
   Download the pre-trained model from the
   [3DUnetCNN](https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/3dunet_dynamic_ndhwc.pb).
   In this example, we are using the model,
   trained using the fold 1 BRATS 2019 data.
   The validation files have been copied from [here](https://github.com/mlcommons/inference/tree/r0.7/vision/medical_imaging/3d-unet/folds)

## 3. Prepare dataset

### Download BraTS 2019 dataset
   Please download [Brats 2019](https://www.med.upenn.edu/cbica/brats2019/data.html)
   separately and unzip the dataset. The directory that contains the dataset files will be
   passed to the launch script when running the benchmarking script.

### Prepare Calibration set
   The calibration set is the forty images listed in brats_cal_images_list.txt. They are randomly selected from Fold 0, Fold 2, Fold 3, and Fold 4 of BraTS 2019 Training Dataset.


# Run command
Please set the following environment variables before running quantization or benchmark commands:

* `export nnUNet_preprocessed=<path/to/build>/build/preprocessed_data`
* `export nnUNet_raw_data_base=<path/to/build>/build/raw_data`
* `export RESULTS_FOLDER=<path/to/build>/build/result`

## Quantization

`bash run_quant.sh --input_model=3dunet_dynamic_ndhwc.pb --dataset_location=<path/to/build>/build --output_model=3dunet_dynamic_ndhwc_int8.pb`

## Benchmark

`bash run_benchmark.sh --input_model=3dunet_dynamic_ndhwc_int8.pb --dataset_location=<path/to/build>/build --batch_size=100 --iters=500 --mode=benchmark`

`bash run_benchmark.sh --input_model=3dunet_dynamic_ndhwc_int8.pb --dataset_location=<path/to/build>/build --batch_size=1 --mode=accuracy`
