Step-by-Step
============

This document is used to list steps of reproducing PyTorch BERT tuning zoo result.
Original BERT documents please refer to [BERT README](../../../../common/README.md) and [README](../../../../common/examples/seq2seq/README.md).

> **Note**
>
> Dynamic Quantization is the recommended method for huggingface models. 

# Prerequisite

## 1. Installation

### Python Version

Recommend python 3.6 or higher version.

#### Install dependency

```shell
pip install -r requirements.txt
```

#### Install PyTorch
```shell
pip install torch
```

## 2. Prepare pretrained model

Before use Intel® Neural Compressor, you should fine tune the model to get pretrained model, You should also install the additional packages required by the examples:

### Seq2seq

#### summarization_billsum task

##### Prepare dataset
```shell
wget https://cdn-datasets.huggingface.co/summarization/pegasus_data/billsum.tar.gz
tar -xzvf billsum.tar.gz
```

# Start to neural_compressor tune for Model Quantization
```shell
cd examples/pytorch/nlp/huggingface_models/summarization/quantization/ptq_dynamic/fx
sh run_tuning.sh --topology=topology_name
```
> NOTE
>
> topology_name can be:{"pegasus_samsum"}
>
> /path/to/summarization/data/dir is the path to data_dir set in finetune.
>
