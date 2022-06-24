Step-by-Step
============

This document is used to list steps of reproducing TensorFlow Intel® Neural Compressor tuning zoo result of 3dunet-mlperf.

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

### 3. Download BraTS 2019 dataset
   Please download [Brats 2019](https://www.med.upenn.edu/cbica/brats2019/data.html)
   separately and unzip the dataset. The directory that contains the dataset files will be
   passed to the launch script when running the benchmarking script.

### 4. Download Pre-trained model
   Download the pre-trained model from the
   [3DUnetCNN](https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/3dunet_dynamic_ndhwc.pb).
   In this example, we are using the model,
   trained using the fold 1 BRATS 2019 data.
   The validation files have been copied from [here](https://github.com/mlcommons/inference/tree/r0.7/vision/medical_imaging/3d-unet/folds)

### 5. Prepare Calibration set
   The calibration set is the forty images listed in brats_cal_images_list.txt. They are randomly selected from Fold 0, Fold 2, Fold 3, and Fold 4 of BraTS 2019 Training Dataset.

### 6. Test command
* `export nnUNet_preprocessed=<path/to/build>/build/preprocessed_data`
* `export nnUNet_raw_data_base=<path/to/build>/build/raw_data`
* `export RESULTS_FOLDER=<path/to/build>/build/result`
* `pip install requirements.txt`
* `python run_accuracy.py --input-model=<path/to/model_file> --data-location=<path/to/dataset> --calib-preprocess=<path/to/calibrationset> --iters=100 --batch-size=1 --mode=benchmark --bfloat16 0`

