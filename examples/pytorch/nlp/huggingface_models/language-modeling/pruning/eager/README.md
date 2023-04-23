Step-by-Step
============

This document presents step-by-step instructions for pruning Huggingface Large Language Models(LLMs) using the Intel® Neural Compressor.


# Prerequisite

## 1. Environment

PyTorch 1.8 or higher version is needed with pytorch_fx backend
The loading of llama models requires transformers version 4.28.0 or higher.


```shell
pip install -r examples/pytorch/nlp/huggingface_models/language-modeling/pruning/eager/requirements.txt
```

## 2. Prepare Dataset

The dataset will be downloaded automatically from the datasets Hub.
See more about loading [huggingface dataset](https://huggingface.co/docs/datasets/loading_datasets.html)


# Run Examples

Intel® Neural Compressor supports pruning and slim operations for LLMs without retraining.Experimentally verified pruning at the MLP layers with channel-wise pattern, which can achieve 10%-20% sparsity and speed up inference while accuracy drop < 1%.
There are pruning scripts for LLM sparse models (GPT-j, BLOOM, OPT, LLaMA etc). The sparse model with can be obtained by modifying pruning parameters. [Pruning Scripts](https://github.com/intel/neural-compressor/tree/master/examples/pytorch/nlp/huggingface_models/language-modeling/pruning/eager/scripts/).


### Results

The pruning performance of GPT-J was verified on different calibration datasets, the last word accuracy data is selected as the maximum of about 5 rounds.
| Model | Calibration dataset | Evaluation dataset | Sparsity pattern | Over mlp block sparsity |Element-wise/matmul, Gemm, conv ratio | Dense last word accuracy | Sparse last word accuracy | Relative drop |
|  :----: | :----: | :----: | :----: | :----: | :----: |:----: |:----:| :----: |
| GPT-J | lambada | lambada | channelx1  | 0.1999 | 0.1242 | 0.7917 | 0.8038 | +1.50% |
| GPT-J | pile_10k | lambada | channelx1  | 0.0999 | 0.0643 | 0.7917 | 0.7925 | +0.10% |
| GPT-J | the_pile | lambada |  channelx1  | 0.0999 | 0.0643 | 0.7917 | 0.7931 | +0.17% |

<br />

The inference performance of the sparse model is verified under different precision on Casual language modeling(CLM) task.
| Model | Task | Calibration dataset | Evaluation dataset | Precision | Dense accuracy | Sparse accuracy | Relative drop |
|  :----: | :----: | :----: | :----: | :----: | :----: |:----: |:----:|
| GPT-J | CLM | pile_10k | lambada_openai | FP32 | 0.6831 | 0.6877 | +0.67% |
| GPT-J | CLM | pile_10k | lambada_openai | IPEX-BF16 | 0.6792 | 0.6833 | +0.60% |

## References
* [A Fast Post-Training Pruning Framework for Transformers](https://arxiv.org/abs/2204.09656)
* [Knowledge Distillation with the Reused Teacher Classifier](https://arxiv.org/abs/2203.14001)

