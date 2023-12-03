Step-by-Step
============

This document is used to list steps of reproducing TensorFlow Intel® Neural Compressor tuning result of Intel® Model Zoo bert large model on squad v1.1 task.
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
For any more details, please follow the procedure in [install-gpu-drivers](https://github.com/intel/intel-extension-for-tensorflow/blob/main/docs/install/install_for_xpu.md#install-gpu-drivers)

#### Quantizing the model on Intel CPU(Optional to install ITEX)
Intel Extension for Tensorflow for Intel CPUs is experimental currently. It's not mandatory for quantizing the model on Intel CPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[cpu]
```
> **Note**: 
> The version compatibility of stock Tensorflow and ITEX can be checked [here](https://github.com/intel/intel-extension-for-tensorflow#compatibility-table). Please make sure you have installed compatible Tensorflow and ITEX.

## 2. Prepare Pretrained model
```shell
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/fp32_bert_squad.pb
```

## 3. Prepare Dataset
Please choose one way to prepare the dataset from the manual approach and the automatic approach.
### Manual approach
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
Run the `prepare_dataset.sh` script located in `examples/tensorflow/nlp/bert_large_squad_model_zoo/quantization/ptq`.

Usage:
```shell
cd examples/tensorflow/nlp/bert_large_squad_model_zoo/quantization/ptq
bash prepare_dataset.sh --output_dir=./data
```

### Convert the dataset to TF Record format
After the dataset is downloaded by either of ways above, the dataset should be converted to files of TF Record format.
```shell
python create_tf_record.py --vocab_file=data/vocab.txt --predict_file=data/dev-v1.1.json --output_file=./eval.tf_record
```

# Run Command
  <b><font color='red'>Please make sure below command should be executed with the same Tensorflow runtime version as above step.</font></b>

## Quantization
  ```shell
  bash run_quant.sh --input_model=./fp32_bert_squad.pb --output_model=./bert_squad_int8.pb --dataset_location=/path/to/evaluation/dataset
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
  ```shell
  bash run_benchmark.sh --input_model=./bert_squad_int8.pb --mode=accuracy --dataset_location=/path/to/evaluation/dataset --batch_size=64
  bash run_benchmark.sh --input_model=./bert_squad_int8.pb --mode=performance --dataset_location=/path/to/evaluation/dataset --batch_size=64
  ```