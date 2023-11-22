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

<br />

# Run Examples

Intel® Neural Compressor provides support for pruning and model slimming operations in Large Language Models (LLMs) without the need for retraining. 

Through experimental verification, it has been observed that pruning the Multi-Layer Perceptron (MLP) layers using a channel-wise pattern can achieve a sparsity level of 10%-20%. This pruning technique speeds up inference while maintaining an accuracy drop of less than 1%. [Retrain-free Example](https://github.com/intel/neural-compressor/tree/master/examples/pytorch/nlp/huggingface_models/language-modeling/pruning/eager/run_clm_no_trainer.py).

The pruning patterns of 1x1 and N:M are supported through the use of the [SparseGPT Example](https://github.com/intel/neural-compressor/tree/master/examples/pytorch/nlp/huggingface_models/language-modeling/pruning/eager/run_clm_sparsegpt.py), It is possible to prune models up to 70B in size within two hours, achieving a sparsity of 40%-50% in both the Multi-Head Attention (MHA) and MLP layers. For models of 7B and above, the drop in accuracy is less than 1%.

Pruning scripts are available for LLM sparse models such as GPT-j, BLOOM, OPT, LLaMA, and the sparse model can be obtained by modifying the pruning parameters. [Pruning Scripts](https://github.com/intel/neural-compressor/tree/master/examples/pytorch/nlp/huggingface_models/language-modeling/pruning/eager/scripts/).

<br />

## Retrain-free Results

The last token accuracy for channel pruning using [the retrain-free scripts](https://github.com/intel/neural-compressor/tree/master/examples/pytorch/nlp/huggingface_models/language-modeling/pruning/eager/scripts/run_gptj_pruning.sh) is presented in the following table.
| Model | Calibration dataset | Evaluation dataset | Sparsity pattern | Over MLP block sparsity |Element-wise/matmul, Gemm, conv ratio | Dense last token accuracy | Sparse last token accuracy | Relative drop |
|  :----: | :----: | :----: | :----: | :----: | :----: |:----: |:----:| :----: |
| EleutherAI/gpt-j-6b | lambada | lambada | channelx1  | 0.1999 | 0.1242 | 0.7917 | 0.8038 | +1.50% |
| EleutherAI/gpt-j-6b | the_pile | lambada |  channelx1  | 0.0999 | 0.0643 | 0.7917 | 0.7931 | +0.17% |
| EleutherAI/gpt-j-6b | pile_10k | lambada | channelx1  | 0.0999 | 0.0643 | 0.7917 | 0.7901 | -0.20% |
| facebook/opt-1.3b | pile_10k | lambada |  channelx1  | 0.0999 | 0.0614 | 0.7541 | 0.7498 | -0.57% |
| facebook/opt-2.7b | pile_10k | lambada |  channelx1  | 0.0999 | 0.0634 | 0.7779 | 0.7778 | -0.01% |
| decapoda-research/llama-7b-hf | pile_10k | lambada |  channelx1  | 0.0999 | 0.0654 | 0.8856 | 0.8815 | -0.46% |
| bigscience/bloom-1b7 | pile_10k | lambada |  channelx1  | 0.0999 | 0.0466 | 0.7143 | 0.7141 | -0.03% |
| bigscience/bloom-7b1 | pile_10k | lambada |  channelx1  | 0.0999 | 0.0568 | 0.7745 | 0.7742 | -0.04% |

<br />

The last word acc of the channel-wise sparse model is shown in the following table. All the sparsity is 10% over MLP block.
| Model | Task | Calibration dataset | Evaluation dataset | Precision | Dense last word accuracy | Sparse last word accuracy | Relative drop |
|  :----: | :----: | :----: | :----: | :----: | :----: |:----: |:----:|
| EleutherAI/gpt-j-6b | CLM | pile_10k | lambada_openai | FP32 | 0.6831 | 0.6819 | -0.17% |
| EleutherAI/gpt-j-6b | CLM | pile_10k | lambada_openai | BF16 | 0.6792 | 0.6767 | -0.36% |
| facebook/opt-1.3b | CLM | pile_10k | lambada_openai | FP32 | 0.5789 |0.5686  | -1.73% |
| facebook/opt-1.3b | CLM | pile_10k | lambada_openai | BF16 | 0.5629 | 0.5501 | -1.78% |
| facebook/opt-2.7b | CLM | pile_10k | lambada_openai | FP32 | 0.6365 | 0.6367 | +0.03% |
| facebook/opt-2.7b | CLM | pile_10k | lambada_openai | BF16 | 0.6336 | 0.6344 | +0.12% |
| decapoda-research/llama-7b-hf | CLM | pile_10k | lambada_openai | FP32 | 0.7361 | 0.7298 | -0.86% |
| decapoda-research/llama-7b-hf | CLM | pile_10k | lambada_openai | BF16 | 0.7326 | 0.7271 | -0.75% |
| bigscience/bloom-1b7 | CLM | pile_10k | lambada_openai | FP32 | 0.4634 | 0.4636 | 0.04% |
| bigscience/bloom-1b7 | CLM | pile_10k | lambada_openai | BF16 | 0.4570 | 0.4572 | 0.04% |
| bigscience/bloom-7b1 | CLM | pile_10k | lambada_openai | FP32 | 0.5764 | 0.5791 | 0.47% |
| bigscience/bloom-7b1 | CLM | pile_10k | lambada_openai | BF16 | 0.5723 | 0.5756 | 0.58% |

<br />

## SparseGPT Results

The last word acc of the 1x1 pattern sparse model using [the sparseGPT script](https://github.com/intel/neural-compressor/tree/master/examples/pytorch/nlp/huggingface_models/language-modeling/pruning/eager/scripts/run_llm_sparsegpt.sh) is shown in the following table.
| Model | Task | Calibration dataset | Evaluation dataset | Sparsity | Precision | Dense last word accuracy | Sparse last word accuracy | Relative drop |
|  :----: | :----: | :----: | :----: | :----: | :----: | :----: |:----: |:----:|
| EleutherAI/gpt-j-6b | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | FP32 | 0.6831 | 0.6922 | +1.33% |
| EleutherAI/gpt-j-6b | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | BF16 | 0.6771 | 0.6874 | +1.52% |
| decapoda-research/llama-7b-hf | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | FP32 | 0.7361 | 0.7332 | -0.39% |
| decapoda-research/llama-7b-hf | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | BF16 | 0.7326 | 0.7297 | -0.23% |
| facebook/opt-6.7b | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | FP32 | 0.6769 | 0.6616 | -2.26% |
| facebook/opt-6.7b | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | BF16 | 0.6730 | 0.6577 | -2.27% |
| tiiuae/falcon-7b | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | FP32 | 0.7467 | 0.7528 | -0.82% |
| tiiuae/falcon-7b | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | BF16 | 0.7464 | 0.7502 | -0.51% |
| bigscience/bloom-7b1 | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | FP32 | 0.5764 | 0.5606 | -2.74% |
| bigscience/bloom-7b1 | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | BF16 | 0.5725 | 0.5587 | -2.41% |
| mosaicml/mpt-7b | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | FP32 | 0.7056 | 0.7035 | -0.30% |
| mosaicml/mpt-7b | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | BF16 | 0.6831 | 0.6856 | +0.37% |
| mosaicml/mpt-7b-chat | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | FP32 | 0.6550 | 0.6561 | +0.17% |
| mosaicml/mpt-7b-chat | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | BF16 | 0.6456 | 0.6451 | -0.08% |
| decapoda-research/llama-13b-hf | CLM | wikitext-2-raw-v1 | lambada_openai | 50% | FP32 | 0.7627 | 0.7559 | -0.89% |
| decapoda-research/llama-13b-hf | CLM | wikitext-2-raw-v1 | lambada_openai | 50% | BF16 | 0.7599 | 0.7559 | -0.53% |


## References

[1] Kwon, W., Kim, S., Mahoney, M.W., Hassoun, J., Keutzer, K. and Gholami, A., 2022. A fast post-training pruning framework for transformers. Advances in Neural Information Processing Systems, 35, pp.24101-24116.

[2] Frantar, E. and Alistarh, D., Sparsegpt: Massive language models can be accurately pruned in one-shot, 2023. URL https://arxiv. org/abs/2301.00774.





