Step-by-Step
============

This document presents step-by-step instructions for pruning Huggingface Large Language Models(LLMs) using the Intel® Neural Compressor.

The retraining free pruning feature is still in development, please stay tuned.

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

Intel® Neural Compressor supports pruning and slimming operations for LLMs without retraining. Experimentally verified pruning at the MLP layers with channel-wise pattern, which can achieve 10%-20% sparsity and speed up inference while accuracy drops < 1%.
There are pruning scripts for LLM sparse models (GPT-j, BLOOM, OPT, LLaMA etc). The sparse model can be obtained by modifying pruning parameters. [Pruning Scripts](https://github.com/intel/neural-compressor/tree/master/examples/pytorch/nlp/huggingface_models/language-modeling/pruning/eager/scripts/).


### Results

The pruning performance of GPT-J was verified on different calibration datasets, the data is selected as the maximum of about 5 rounds.
| Model | Calibration dataset | Evaluation dataset | Sparsity pattern | Over MLP block sparsity |Element-wise/matmul, Gemm, conv ratio | Dense last token accuracy | Sparse last token accuracy | Relative drop |
|  :----: | :----: | :----: | :----: | :----: | :----: |:----: |:----:| :----: |
| EleutherAI/gpt-j-6b | lambada | lambada | channelx1  | 0.1999 | 0.1242 | 0.7917 | 0.8038 | +1.50% |
| EleutherAI/gpt-j-6b | pile_10k | lambada | channelx1  | 0.0999 | 0.0643 | 0.7917 | 0.7925 | +0.10% |
|EleutherAI/gpt-j-6b | the_pile | lambada |  channelx1  | 0.0999 | 0.0643 | 0.7917 | 0.7931 | +0.17% |

<br />

The last word acc of the sparse model is verified under different precision on Casual language modeling(CLM) task. All the sparsity is 10% over MLP block.
| Model | Task | Calibration dataset | Evaluation dataset | Precision | Dense accuracy | Sparse accuracy | Relative drop |
|  :----: | :----: | :----: | :----: | :----: | :----: |:----: |:----:|
| EleutherAI/gpt-j-6b | CLM | pile_10k | lambada_openai | FP32 | 0.6831 | 0.6877 | +0.67% |
| EleutherAI/gpt-j-6b | CLM | pile_10k | lambada_openai | IPEX-BF16 | 0.6792 | 0.6833 | +0.60% |
| facebook/opt-2.7b | CLM | pile_10k | lambada_openai | FP32 | 0.6365 |0.6367  | +0.03% |
| facebook/opt-2.7b | CLM | pile_10k | lambada_openai | IPEX-BF16 | 0.6336| 0.6344 | +0.12% |
| facebook/opt-1.3b | CLM | pile_10k | lambada_openai | FP32 | 0.5789 |0.5686  | -1.73% |
| facebook/opt-1.3b | CLM | pile_10k | lambada_openai | IPEX-BF16 | 0.5629| 0.5501 | -1.78% |

## References
* [A Fast Post-Training Pruning Framework for Transformers](https://arxiv.org/abs/2204.09656)



