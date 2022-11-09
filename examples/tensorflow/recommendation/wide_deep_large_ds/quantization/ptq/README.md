Step-by-Step
============

This document is used to list steps of reproducing TensorFlow Wide & Deep tuning zoo result.
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

### 4. Install Additional Dependency packages
```shell
cd examples/tensorflow/recommendation/wide_deep_large_ds/quantization/ptq
pip install -r requirements.txt
```

### 5. Prepare Dataset
Download training dataset: (8 million samples)
```bash
$ wget https://storage.googleapis.com/dataset-uploader/criteo-kaggle/large_version/train.csv
```
Download evaluation dataset (2 million samples)
```bash
$ wget https://storage.googleapis.com/dataset-uploader/criteo-kaggle/large_version/eval.csv
```

### 6. Process Dataset
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

### 7. Download Frozen PB
```shell
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/wide_deep_fp32_pretrained_model.pb
```

### 8. Config the yaml file
In examples directory, there is a wide_deep_large_ds.yaml for tuning the model on Intel CPUs. The 'framework' in the yaml is set to 'tensorflow'. If running this example on Intel GPUs, the 'framework' should be set to 'tensorflow_itex' and the device in yaml file should be set to 'gpu'. The wide_deep_large_ds_itex.yaml is prepared for the GPU case. We could remove most of items and only keep mandatory item for tuning. We also implement a calibration dataloader and have evaluation field for creation of evaluation function at internal neural_compressor.

### 9. Run Command
  # The cmd of running WnD
  ```shell
  bash run_tuning.sh --dataset_location=/path/to/datasets --input_model=/path/to/wide_deep_fp32_pretrained_model.pb --output_model=./wnd_int8_opt.pb
  bash run_benchmark.sh --dataset_location=/path/to/datasets --input_model=./wnd_int8_opt.pb --mode=accuracy --batch_size=500
  bash run_benchmark.sh --dataset_location=/path/to/datasets --input_model=./wnd_int8_opt.pb --mode=benchmark --batch_size=500
  ```
### Other
This example takes the reference from https://github.com/IntelAI/models/tree/master/benchmarks/recommendation/tensorflow/wide_deep_large_ds.
The pretrained model was trained with preprocessed data from dataset Criteo.
