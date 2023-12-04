Step-by-Step
============

This document is used to list steps of reproducing TensorFlow Wide & Deep tuning zoo result.
This example can run on Intel CPUs and GPUs.

# Prerequisite

## 1. Environment

### Installation
```shell
# Install Intel® Neural Compressor
pip install neural-compressor
```
### Install Intel Tensorflow
```shell
pip install intel-tensorflow
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

### Install Additional Dependency packages
```shell
cd examples/tensorflow/recommendation/wide_deep_large_ds/quantization/ptq
pip install -r requirements.txt
```

### 2. Download Frozen PB
```shell
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/wide_deep_fp32_pretrained_model.pb
```

### 3. Prepare Dataset
Download training dataset: (8 million samples)
```bash
$ wget https://storage.googleapis.com/dataset-uploader/criteo-kaggle/large_version/train.csv
```
Download evaluation dataset (2 million samples)
```bash
$ wget https://storage.googleapis.com/dataset-uploader/criteo-kaggle/large_version/eval.csv
```

### 4. Process Dataset
Process calib dataset
```bash
python preprocess_csv_tfrecords.py \
        --inputcsv-datafile train.csv \
        --calibrationcsv-datafile eval.csv \
        --outputfile-name processed_data
```
Process eval dataset
```bash
python preprocess_csv_tfrecords.py \
        --inputcsv-datafile eval.csv \
        --calibrationcsv-datafile train.csv \
        --outputfile-name processed_data
```
Two .tfrecords files are generated and will be used later on:
1) train_processed_data.tfrecords
2) eval_processed_data.tfrecords


# Run Command

## Quantization
  ```shell
  bash run_quant.sh --dataset_location=/path/to/datasets --input_model=/path/to/wide_deep_fp32_pretrained_model.pb --output_model=./wnd_int8_opt.pb
  ```

### Quantization Config
The Quantization Config class has default parameters setting for running on Intel CPUs. If running this example on Intel GPUs, the 'backend' parameter should be set to 'itex' and the 'device' parameter should be set to 'gpu'.

```
config = PostTrainingQuantConfig(
        device="gpu",
        backend="itex",
        ...
        )
```

## Benchmark
  ```
  bash run_benchmark.sh --dataset_location=/path/to/datasets --input_model=./wnd_int8_opt.pb --mode=accuracy --batch_size=500
  bash run_benchmark.sh --dataset_location=/path/to/datasets --input_model=./wnd_int8_opt.pb --mode=performance --batch_size=500
  ```

# Other
This example takes the reference from https://github.com/IntelAI/models/tree/master/benchmarks/recommendation/tensorflow/wide_deep_large_ds.
The pretrained model was trained with preprocessed data from dataset Criteo.
