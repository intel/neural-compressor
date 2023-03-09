Step-by-Step
============

This document presents step-by-step instructions for pruning Huggingface models on translation tasks using the Intel® Neural Compressor. It also provides an example of flan-T5-small pruning.

# Prerequisite

## 1. Environment

PyTorch 1.8 or higher version is needed with pytorch_fx backend.

```shell
pip install -r examples/pytorch/nlp/huggingface_models/question-answering/pruning/eager/requirements.txt
```

## 2. Prepare Dataset

The dataset will be downloaded automatically from the datasets Hub.
See more about loading [huggingface dataset](https://huggingface.co/docs/datasets/loading_datasets.html)


# Run Examples
One example of pruning a Flan-T5-small model is provided, which is trained on wmt16 English-Romanian task. We are working on providing more pruning examples and sharing our sparse models on HuggingFace.



### Results
The snip-momentum pruning method is used by default and the initial dense model us fine-tuned.
|  Model  | Dataset  |  Sparsity pattern | Element-wise/matmul, Gemm, conv ratio | Dense BLEU | Sparse BLEU | Relative drop|
|  :----:  | :----:  | :----: | :----: |:----: |:----:| :----: |
| Flan-T5-small | wmt16 en-ro |  4x1  | 0.7993 | 25.63 | 24.35 | -4.95% |


## References
* [SNIP: Single-shot Network Pruning based on Connection Sensitivity](https://arxiv.org/abs/1810.02340)
* [Knowledge Distillation with the Reused Teacher Classifier](https://arxiv.org/abs/2203.14001)





