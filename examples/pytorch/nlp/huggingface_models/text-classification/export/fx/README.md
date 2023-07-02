Step-by-Step
============

This document is used to introduce the steps of exporting PyTorch model into ONNX format.

# Prerequisite

## 1. Installation

### Python Version

Recommend python 3.6 or higher version.

#### Install Transformers

```bash
pip install transformers
```

#### Install dependency

```shell
pip install -r requirements.txt
```

#### Install PyTorch
```shell
pip install torch
```

## 2. Prepare pretrained model

Before using Intel® Neural Compressor, you should fine tune the model to get pretrained models or reuse fine-tuned models in [model hub](https://huggingface.co/models). You should extraly install the complementary packages required by the examples.

## 3. Prepare dataset

Please pass in the name of dataset, supported datasets are 'mrpc', 'qqp', 'qnli', 'rte', 'sts-b', 'cola', 'mnli', 'wnli', 'sst2'.


# Run

### 1. Get the exported model: 

```bash
# export fp32 model
bash run_export.sh --input_model=[model_name_or_path] --dataset_location=[dataset_name] --dtype=fp32 --output_model=bert-fp32.onnx
# export int8 model
bash run_export.sh --input_model=[model_name_or_path] --dataset_location=[dataset_name]  --dtype=int8 --quant_format=[QDQ/QOperator] --output_model=bert-int8.onnx --approach=[static|dynamic]
``` 

### 2. Get the benchmark results of exported and tuned models, including Batch_size and Throughput: 
```bash
# benchmark ONNX model
bash run_benchmark.sh --input_model=[bert-fp32.onnx|bert-int8.onnx] --dataset_location=[dataset_name] --tokenizer=[model_name_or_path] --mode=[accuracy|performance] --batch_size=[16]
# benchmark PyTorch model
bash run_benchmark.sh --input_model=[model_name_or_path|/path/to/saved_results] --dataset_location=[dataset_name] --mode=[accuracy|performance] --int8=[true|false] --batch_size=[16]
```
