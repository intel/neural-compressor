Step-by-Step
============

This document is used to list steps of reproducing TensorFlow Wide & Deep tuning zoo result.

## Prerequisite

### 1. Installation
```shell
# Install Intel® Low Precision Optimization Tool
pip install lpot
```
### 2. Install Intel Tensorflow
```shell
pip install intel-tensorflow
```
> Note: Supported Tensorflow [Version](../../../../README.md).

### 3. Install Additional Dependency packages
```shell
cd examples/tensorflow/object_detection && pip install -r requirements.txt
```

### 4. Prepare Dataset
Download training dataset: (8 million samples)
```bash
$ wget https://storage.googleapis.com/dataset-uploader/criteo-kaggle/large_version/train.csv
```
Download evaluation dataset (2 million samples)
```bash
$ wget https://storage.googleapis.com/dataset-uploader/criteo-kaggle/large_version/eval.csv
```

### 5. Process Dataset
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

### 6. Download Frozen PB
```shell
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/wide_deep_fp32_pretrained_model.pb
```

### 7. Run Command
  # The cmd of running WnD
  ```shell
  bash run_tuning.sh --dataset_location=/path/to/datasets --input_model=/path/to/wide_deep_fp32_pretrained_model.pb --output_model=./wnd_int8_opt.pb
  bash run_benchmark.sh --dataset_location=/path/to/datasets --input_model=./wnd_int8_opt.pb --mode=accuracy --batch_size=500
  bash run_benchmark.sh --dataset_location=/path/to/datasets --input_model=./wnd_int8_opt.pb --mode=benchmark --batch_size=500
  ```
### Other
This example takes the reference from https://github.com/IntelAI/models/tree/master/benchmarks/recommendation/tensorflow/wide_deep_large_ds.
The pretrained model was trained with preprocessed data from dataset Criteo.
