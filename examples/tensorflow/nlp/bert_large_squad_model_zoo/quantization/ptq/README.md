Step-by-Step
============

This document is used to list steps of reproducing TensorFlow Intel® Neural Compressor tuning result of Intel® Model Zoo bert large model on squad v1.1 task.


## Prerequisite

### 1. Installation
```shell
# Install Intel® Neural Compressor
pip install neural-compressor
```
### 2. Install Intel Tensorflow
```python
pip install intel-tensorflow
```

### 3. Prepare Dataset
```shell
wget https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip
```

```shell
unzip wwm_uncased_L-24_H-1024_A-16.zip
```

```shell
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -P wwm_uncased_L-24_H-1024_A-16
```
wwm_uncased_L-24_H-1024_A-16 folder will be located on your data path.

#### Automatic dataset download
Run the `prepare_dataset.sh` script located in `examples/tensorflow/nlp/bert_large_squad/quantization/ptq`.

Usage:
```shell
cd examples/tensorflow/nlp/bert_large_squad/quantization/ptq
bash prepare_dataset.sh --output_dir=./data
```

Then create the tf_record file and you need to config the tf_record path in yaml file.
```shell
python create_tf_record.py --vocab_file=data/vocab.txt --predict_file=data/dev-v1.1.json --output_file=./eval.tf_record
```

### 4. Prepare Pretrained model
```shell
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/fp32_bert_squad.pb
```

## Run Command
  <b><font color='red'>Please make sure below command should be executed with the same Tensorflow runtime version as above step.</font></b>

### Run Tuning
  ```shell
  python tune_squad.py --config=./bert.yaml --input_model=./bert_fp32.pb --output_model=./int8.pb --tune
  ```

### Run Benchmark
  ```shell
  python tune_squad.py --config=./bert.yaml --input_model=./int8.pb --benchmark
  ```