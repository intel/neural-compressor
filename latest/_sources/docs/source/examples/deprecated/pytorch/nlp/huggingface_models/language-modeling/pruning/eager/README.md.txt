Step-by-Step
============

This document presents step-by-step instructions for pruning Huggingface Large Language Models(LLMs) using the Intel® Neural Compressor.

The retraining free pruning feature is still in development, please stay tuned.

# Prerequisite

## 1. Environment

PyTorch 1.8 or higher version is needed with pytorch_fx backend
The transformers version required varies across different types of models. Here, the transformers version used for running models during experiments is provided as a reference.
| Model | Transformers version |
|  :----: | :----: |
| EleutherAI/gpt-j-6b | 4.28/4.30/4.34/4.36 |
| huggyllama/llama-7b | 4.28/4.30/4.34/4.36 |
| meta-llama/Llama-2-7b-hf | 4.30/4.34/4.36 |
| facebook/opt-6.7b | 4.28/4.30/4.34/4.36 |
| databricks/dolly-v2-3b | 4.28/4.30/4.34/4.36 |
| tiiuae/falcon-7b | 4.28/4.30/4.34/4.36 |
| mosaicml/mpt-7b | 4.28/4.30/4.34/4.36 |
| bigscience/bloom-7b1 | 4.28/4.30/4.34/4.36 |
| baichuan-inc/Baichuan-7B | 4.28/4.30 |
| Qwen/Qwen-7B | 4.28/4.30/4.34/4.36 |
| THUDM/chatglm3-6b | 4.34/4.36 |
| mistralai/Mistral-7B-v0.1 | 4.34/4.36 |


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

The pruning patterns of 1x1 and N:M are supported through the use of the [SparseGPT Example](https://github.com/intel/neural-compressor/tree/master/examples/pytorch/nlp/huggingface_models/language-modeling/pruning/eager/run_clm_sparsegpt.py), It is possible to prune models up to 70B in size within two hours, achieving a sparsity of 40%-60% in both the Multi-Head Attention (MHA) and MLP layers. For models of 7B and above, the drop in accuracy is less than 1%.
Note that Pruning for models at 30 billion parameters and above can be done on a single GPU card (such as the A100), while evaluation is recommended to be performed using multiple cards:
```shell
    CUDA_VISIBLE_DEVICES=0,1 \
    python examples/pytorch/nlp/huggingface_models/language-modeling/pruning/eager/run_clm_sparsegpt.py \
    --model_name_or_path /PATH/TO/SPARSE/LLM/ \
    --device=0 \
    --eval_dtype 'bf16' \
    --per_device_eval_batch_size 2
```

Pruning scripts are available for LLM sparse models such as GPT-j, BLOOM, OPT, LLaMA, Qwen, Chatglm, Mpt, Falcon, and the sparse model can be obtained by modifying the pruning parameters. [Pruning Scripts](https://github.com/intel/neural-compressor/tree/master/examples/pytorch/nlp/huggingface_models/language-modeling/pruning/eager/scripts/).

<br />

## Retrain-free Results

The last token accuracy for channel pruning using [the retrain-free scripts](https://github.com/intel/neural-compressor/tree/master/examples/pytorch/nlp/huggingface_models/language-modeling/pruning/eager/scripts/run_gptj_pruning.sh) is presented in the following table.

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


## SparseGPT Results

The last word acc of the 1x1 pattern sparse model using [the sparseGPT script](https://github.com/intel/neural-compressor/tree/master/examples/pytorch/nlp/huggingface_models/language-modeling/pruning/eager/scripts/run_llm_sparsegpt.sh) is shown in the following table.


| Model | Task | Calibration dataset | Evaluation dataset | Sparsity | Precision | Dense last word accuracy | Sparse last word accuracy | Relative drop |
|  :----: | :----: | :----: | :----: | :----: | :----: | :----: |:----: |:----:|
| EleutherAI/gpt-j-6b | CLM | NeelNanda/pile-10k | lambada_openai | 40% | FP32 | 0.6831 | 0.6922 | +2.30% |
| EleutherAI/gpt-j-6b | CLM | NeelNanda/pile-10k | lambada_openai | 40% | BF16 | 0.6781 | 0.6874 | +1.48% |
| meta-llama/Llama-2-7b-hf | CLM | NeelNanda/pile-10k | lambada_openai | 40% | FP32 | 0.7392 | 0.7411 | +0.26% |
| meta-llama/Llama-2-7b-hf | CLM | NeelNanda/pile-10k | lambada_openai | 40% | BF16 | 0.7361 | 0.7376 | -0.22% |
| huggyllama/llama-7b | CLM | NeelNanda/pile-10k | lambada_openai | 40% | FP32 | 0.7361 | 0.7450 | +1.21% |
| huggyllama/llama-7b | CLM | NeelNanda/pile-10k | lambada_openai | 40% | BF16 | 0.7308 | 0.7427 | +0.90% |
| facebook/opt-6.7b | CLM | NeelNanda/pile-10k | lambada_openai | 40% | FP32 | 0.6769 | 0.6897 | +1.89% |
| facebook/opt-6.7b | CLM | NeelNanda/pile-10k | lambada_openai | 40% | BF16 | 0.6765 | 0.6856 | +1.29% |
| tiiuae/falcon-7b | CLM | NeelNanda/pile-10k | lambada_openai | 40% | FP32 | 0.7467 | 0.7555 | +1.18% |
| tiiuae/falcon-7b | CLM | NeelNanda/pile-10k | lambada_openai | 40% | BF16 | 0.7467 | 0.7561 | +1.26% |
| bigscience/bloom-7b1 | CLM | NeelNanda/pile-10k | lambada_openai | 40% | FP32 | 0.5764 | 0.5768 | +0.07% |
| bigscience/bloom-7b1 | CLM | NeelNanda/pile-10k | lambada_openai | 40% | BF16 | 0.5731 | 0.5738 | -0.45% |
| mosaicml/mpt-7b | CLM | NeelNanda/pile-10k | lambada_openai | 40% | FP32 | 0.7056 | 0.7114 | +0.82% |
| mosaicml/mpt-7b | CLM | NeelNanda/pile-10k | lambada_openai | 40% | BF16 | 0.6831 | 0.6920 | -1.93% |
| THUDM/chatglm3-6b | CLM | NeelNanda/pile-10k | lambada_openai | 40% | FP32 | 0.5888 | 0.5822 | -1.12% |
| THUDM/chatglm3-6b | CLM | NeelNanda/pile-10k | lambada_openai | 40% | BF16 | 0.5878 | 0.5812 | -1.29% |
| mistralai/Mistral-7B-v0.1 | CLM | NeelNanda/pile-10k | lambada_openai | 40% | FP32 | 0.7590 | 0.7803 | +2.81% |
| mistralai/Mistral-7B-v0.1 | CLM | NeelNanda/pile-10k | lambada_openai | 40% | BF16 | 0.7561 | 0.7770 | +2.37% |
| Qwen/Qwen-7B | CLM | NeelNanda/pile-10k | lambada_openai | 40% | FP32 | 0.6996 | 0.7085 | +1.27% |
| Qwen/Qwen-7B | CLM | NeelNanda/pile-10k | lambada_openai | 40% | BF16 | 0.6959 | 0.7077 | +1.16% |
| mosaicml/mpt-7b-chat | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | FP32 | 0.6550 | 0.6561 | +0.17% |
| mosaicml/mpt-7b-chat | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | BF16 | 0.6456 | 0.6451 | -1.51% |
| meta-llama/Llama-2-13b-hf | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | FP32 | 0.7679 | 0.7629 | -0.65% |
| meta-llama/Llama-2-13b-hf | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | BF16 | 0.7667 | 0.7601 | -1.02% |
| decapoda-research/llama-13b-hf | CLM | wikitext-2-raw-v1 | lambada_openai | 50% | FP32 | 0.7627 | 0.7559 | -0.89% |
| decapoda-research/llama-13b-hf | CLM | wikitext-2-raw-v1 | lambada_openai | 50% | BF16 | 0.7599 | 0.7559 | -0.89% |
| meta-llama/Llama-2-70b-hf | CLM | wikitext-2-raw-v1 | lambada_openai | 60% | FP32 | 0.7964 | 0.7951 | -0.16% |
| meta-llama/Llama-2-70b-hf | CLM | wikitext-2-raw-v1 | lambada_openai | 60% | BF16 | 0.7937 | 0.7943 | -0.26% |
| Qwen/Qwen-72B | CLM | wikitext-2-raw-v1 | lambada_openai | 60% | FP32 | 0.7702 | 0.7859 | +2.04% |
| Qwen/Qwen-72B | CLM | wikitext-2-raw-v1 | lambada_openai | 60% | BF16 | 0.7673 | 0.7813 | +1.44% |

<!-- discarded data -->
<!-- | meta-llama/Llama-2-7b-hf | CLM | wikitext-2-raw-v1 | lambada_openai | 30% | FP32 | 0.7392 | 0.7320 | -0.97% |
| meta-llama/Llama-2-7b-hf | CLM | wikitext-2-raw-v1 | lambada_openai | 30% | BF16 | 0.7361 | 0.7304 | -1.19% |
| EleutherAI/gpt-j-6b | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | FP32 | 0.6831 | 0.6922 | +1.33% |
| EleutherAI/gpt-j-6b | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | BF16 | 0.6781 | 0.6874 | +0.63% |
| huggyllama/llama-7b | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | FP32 | 0.7361 | 0.7332 | -0.39% |
| huggyllama/llama-7b | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | BF16 | 0.7308 | 0.7297 | -0.87% |
| facebook/opt-6.7b | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | FP32 | 0.6769 | 0.6616 | -2.26% |
| facebook/opt-6.7b | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | BF16 | 0.6765 | 0.6577 | -2.84% |
| tiiuae/falcon-7b | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | FP32 | 0.7467 | 0.7528 | +0.82% |
| tiiuae/falcon-7b | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | BF16 | 0.7467 | 0.7502 | +0.47% |
| bigscience/bloom-7b1 | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | FP32 | 0.5764 | 0.5606 | -2.74% |
| bigscience/bloom-7b1 | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | BF16 | 0.5731 | 0.5587 | -3.07% |
| mosaicml/mpt-7b | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | FP32 | 0.7056 | 0.7035 | -0.30% |
| mosaicml/mpt-7b | CLM | wikitext-2-raw-v1 | lambada_openai | 40% | BF16 | 0.6831 | 0.6839 | -2.83% | -->

## References

[1] Kwon, W., Kim, S., Mahoney, M.W., Hassoun, J., Keutzer, K. and Gholami, A., 2022. A fast post-training pruning framework for transformers. Advances in Neural Information Processing Systems, 35, pp.24101-24116.

[2] Frantar, E. and Alistarh, D., 2023, July. Sparsegpt: Massive language models can be accurately pruned in one-shot. In International Conference on Machine Learning (pp. 10323-10337). PMLR.




