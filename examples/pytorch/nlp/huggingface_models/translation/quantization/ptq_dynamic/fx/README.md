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

# Start to neural_compressor tune for Model Quantization
```shell
cd examples/pytorch/nlp/huggingface_models/translation/quantization/ptq_dynamic/fx
sh run_tuning.sh --topology=topology_name
```
> NOTE
>
> topology_name can be:{"t5_WMT_en_ro", "marianmt_WMT_en_ro"}
>
