Step-by-Step
============

This document presents step-by-step instructions for pruning Huggingface models using the Intel® Neural Compressor.

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
Several pruning examples are provided, which are trained on different datasets/tasks, use different sparsity patterns, etc. We are working on sharing our sparse models on HuggingFace.

There are pruning scripts for SQuAD sparse models (Bert-mini, Distilbert-base-uncased, Bert-base-uncased, Bert-large, etc). The sparse model with different patterns ("4x1", "2:4", etc) can be obtained by modifying "target_sparsity" and "pruning_pattern" parameters. [Pruning Scripts](https://github.com/intel/neural-compressor/tree/master/examples/pytorch/nlp/huggingface_models/question-answering/pruning/eager/scripts/).

Fine-tuning of the dense model is also supported (by setting --do_prune to False) [Bert-mini SQuAD](https://github.com/intel/neural-compressor/tree/master/examples/pytorch/nlp/huggingface_models/question-answering/pruning/eager/scripts/bertmini_dense_fintune.sh)


### Results
The snip-momentum pruning method is used by default and the initial dense models are all fine-tuned.
|  Model  | Dataset  |  Sparsity pattern | Element-wise/matmul, Gemm, conv ratio | Dense F1 (mean/max)| Sparse F1 (mean/max)| Relative drop|
|  :----:  | :----:  | :----: | :----: |:----: |:----:| :----: |
| Bert-mini | SQuAD |  4x1  | 0.7993 | 0.7662/0.7687 | 0.7617/0.7627 | -0.78% |
| Bert-mini | SQuAD |  2:4  | 0.4795 | 0.7662/0.7687 | 0.7733/0.7762 | +0.98% |
| Distilbert-base-uncased | SQuAD |  4x1  | 0.7986 | 0.8690 | 0.8615 | -0.86% |
| Distilbert-base-uncased | SQuAD |  2:4  | 0.5000 | 0.8690 | 0.8731/0.8750 | +0.69% |
| Bert-base-uncased | SQuAD |  4x1  | 0.7986 | 0.8859 | 0.8778 | -0.92% |
| Bert-base-uncased | SQuAD |  2:4  | 0.5000 | 0.8859 | 0.8924/0.8940 | +0.91% |
| Bert-large | SQuAD |  4x1  | 0.7988 | 0.9123 | 0.9091 | -0.35% |
| Bert-large | SQuAD |  2:4  | 0.5002 | 0.9123 | 0.9167 | +0.48% |

## References
* [SNIP: Single-shot Network Pruning based on Connection Sensitivity](https://arxiv.org/abs/1810.02340)
* [Knowledge Distillation with the Reused Teacher Classifier](https://arxiv.org/abs/2203.14001)




