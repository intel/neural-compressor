Step-by-Step
============

This document is used to list steps of reproducing TensorFlow Intel® Neural Compressor tuning result of Intel® Model Zoo bert large model on squad v1.1 task.
This example can run on Intel CPUs and GPUs.


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

### 4. Prepare Dataset
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

### 5. Prepare Pretrained model
```shell
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/fp32_bert_squad.pb
```

## Write Yaml config file
In examples directory, there is a bert.yaml for tuning the model on Intel CPUs. The 'framework' in the yaml is set to 'tensorflow'. If running this example on Intel GPUs, the 'framework' should be set to 'tensorflow_itex' and the device in yaml file should be set to 'gpu'. The bert_itex.yaml is prepared for the GPU case. We could remove most of items and only keep mandatory item for tuning. We also implement a calibration dataloader and have evaluation field for creation of evaluation function at internal neural_compressor.

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