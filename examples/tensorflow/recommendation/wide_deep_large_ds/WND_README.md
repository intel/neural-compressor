Step-by-Step
============

This document is used to list steps of reproducing TensorFlow Wide & Deep iLiT tuning zoo result.


## Prerequisite

### 1. Installation
```Shell
# Install iLiT
pip instal ilit
```
### 2. Install Intel Tensorflow 1.15/2.0/2.1
```shell
pip intel-tensorflow==1.15.2 [2.0,2.1]
```

### 3. Install Additional Dependency packages
```shell
cd examples/tensorflow/object_detection && pip install -r requirements.txt
```

### 4. Prepare Dataset
Download training dataset: (8 million samples)
```
$ wget https://storage.googleapis.com/dataset-uploader/criteo-kaggle/large_version/train.csv
```
Download evaluation dataset (2 million samples)
```
$ wget https://storage.googleapis.com/dataset-uploader/criteo-kaggle/large_version/eval.csv
```

### 5. Process Dataset
Process calib dataset
```
python preprocess_csv_tfrecords.py \
        --inputcsv-datafile train.csv \
        --calibrationcsv-datafile train.csv \
        --outputfile-name preprocessed_data
```
Process eval dataset
```
python preprocess_csv_tfrecords.py \
        --inputcsv-datafile eval.csv \
        --calibrationcsv-datafile train.csv \
        --outputfile-name preprocessed_data
```
Two .tfrecords files are generated and will be used later on:
1) train_preprocessed_data.tfrecords
2) eval_preprocessed_data.tfrecords

### 6. Download Frozen PB
```shell
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/wide_deep_fp32_pretrained_model.pb
```

## Run Command
  # The cmd of running WnD
  ```Shell
  bash run_tuning.sh    --dataset_location=/path/to/datasets  --input_model=/path/to/wide_deep_fp32_pretrained_model.pb --output_model=./wnd_int8_opt.pb
  bash run_benchmark.sh --dataset_location=/path/to/datasets --input_model=./wnd_int8_opt.pb --mode=accuracy --batch_size=500
  bash run_benchmark.sh --dataset_location=/path/to/datasets --input_model=./wnd_int8_opt.pb --mode=benchmark --batch_size=500
  ```



