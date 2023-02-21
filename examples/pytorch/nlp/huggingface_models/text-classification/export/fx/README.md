Step-by-Step
============

This document is used to list steps of reproducing PyTorch BERT tuning zoo result.
Original BERT documents please refer to [BERT README](../../../../common/README.md) and [README](../../../../common/examples/text-classification/README.md).

> **Note**
>
> Dynamic Quantization is the recommended method for huggingface models. 

# Prerequisite

## 1. Installation

### Python Version

Recommend python 3.6 or higher version.

#### Install BERT model

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

Before use Intel® Neural Compressor, you should fine tune the model to get pretrained model or reuse fine-tuned models in [model hub](https://huggingface.co/models), You should also install the additional packages required by the examples.


# Run

### 1. To get the exported model: 

```bash
# export fp32 model
bash run_export.sh --input_model=[model_name_or_path] --dataset_location=mrpc --dtype=fp32
# export int8 model
bash run_export.sh --input_model=[model_name_or_path] --dataset_location=mrpc --dtype=int8 --quant_format=[QDQ/QLinear]
``` 

### 2. To get the benchmark of exported and tuned models, includes Batch_size and Throughput: 
```bash
# benchmark ONNX model
bash run_benchmark.sh --input_model=[fp32-model.onnx|int8-QDQ-model.onnx|int8-QLinear-model.onnx] --dataset_location=/path/to/onnx-imagenet-validation --tokenizer=[model_name_or_path] --mode=[accuracy|performance] --batch_size=[16]
# benchmark PyTorch model
bash run_benchmark.sh --input_model=[model_name_or_path|/path/to/saved_results] --dataset_location=/path/to/pytorch-imagenet --mode=[accuracy|performance] --int8=[true|false] --batch_size=[16]
```
