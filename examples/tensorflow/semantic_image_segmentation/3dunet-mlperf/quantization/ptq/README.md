Step-by-Step
============

This document is used to list steps of reproducing TensorFlow Intel® Neural Compressor tuning zoo result of 3dunet-mlperf.
This example can run on Intel CPUs and GPUs.

## Prerequisite

### 1. Installation
```shell
# Install Intel® Neural Compressor
pip install neural-compressor
```

### 2. Install Intel Tensorflow
```shell
pip install intel-tensorflow
```
> Note: Supported Tensorflow [Version](../../../../../../README.md#supported-frameworks).

### 3. Install Intel Extension for Tensorflow
#### Quantizing the model on Intel GPU
Intel Extension for Tensorflow is mandatory to be installed for quantizing the model on Intel GPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[gpu]
```
For any more details, please follow the procedure in [install-gpu-drivers](https://github.com/intel-innersource/frameworks.ai.infrastructure.intel-extension-for-tensorflow.intel-extension-for-tensorflow/blob/master/docs/install/install_for_gpu.md#install-gpu-drivers)

#### Quantizing the model on Intel CPU(Experimental)
Intel Extension for Tensorflow for Intel CPUs is experimental currently. It's not mandatory for quantizing the model on Intel CPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[cpu]
```

### 4. Download BraTS 2019 dataset
   Please download [Brats 2019](https://www.med.upenn.edu/cbica/brats2019/data.html)
   separately and unzip the dataset. The directory that contains the dataset files will be
   passed to the launch script when running the benchmarking script.

### 5. Download Pre-trained model
   Download the pre-trained model from the
   [3DUnetCNN](https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/3dunet_dynamic_ndhwc.pb).
   In this example, we are using the model,
   trained using the fold 1 BRATS 2019 data.
   The validation files have been copied from [here](https://github.com/mlcommons/inference/tree/r0.7/vision/medical_imaging/3d-unet/folds)

### 6. Prepare Calibration set
   The calibration set is the forty images listed in brats_cal_images_list.txt. They are randomly selected from Fold 0, Fold 2, Fold 3, and Fold 4 of BraTS 2019 Training Dataset.

### 7. Config the yaml file
In examples directory, there is a 3dunet-mlperf.yaml for tuning the model on Intel CPUs. The 'framework' in the yaml is set to 'tensorflow'. If running this example on Intel GPUs, the 'framework' should be set to 'tensorflow_itex' and the device in yaml file should be set to 'gpu'. The 3dunet-mlperf_itex.yaml is prepared for the GPU case. We could remove most of items and only keep mandatory item for tuning. We also implement a calibration dataloader and have evaluation field for creation of evaluation function at internal neural_compressor.

### 8. Test command
* `export nnUNet_preprocessed=<path/to/build>/build/preprocessed_data`
* `export nnUNet_raw_data_base=<path/to/build>/build/raw_data`
* `export RESULTS_FOLDER=<path/to/build>/build/result`
* `pip install requirements.txt`
* `python run_accuracy.py --input-model=<path/to/model_file> --data-location=<path/to/dataset> --calib-preprocess=<path/to/calibrationset> --iters=100 --batch-size=1 --mode=benchmark --bfloat16 0`

