Step-by-Step
============
This document describes the step-by-step instructions to run large language models (LLMs) on 4th Gen Intel® Xeon® Scalable Processor (codenamed Sapphire Rapids) with PyTorch and Intel® Extension for PyTorch.

The script `run_clm_no_trainer.py` supports `GPTJ`, `OPT`, `LLaMA2`, `BLOOM` and `Falcon` quantization and validates last word prediction accuracy with [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness.git) now, and we are adding more models.

# Prerequisite
## 1. Create Environment
```
# Installation
pip install -r requirements.txt
```

# Run

Here is how to run the scripts:

**Causal Language Modeling (CLM)**

`run_clm_no_trainer.py` quantizes the large language models using the dataset [NeelNanda/pile-10k](https://huggingface.co/datasets/NeelNanda/pile-10k) calibration and validates `lambada_openai`, `piqa`, `winogrande`, `hellaswag` and other datasets accuracy provided by lm_eval, an example command is as follows.
### GPT-J-6b

#### Quantization
```bash
# "--approach weight_only" is used to enable weight only quantization.
# RTN
python run_clm_no_trainer.py \
    --model EleutherAI/gpt-j-6B \
    --quantize \
    --approach weight_only \
    --woq_bits 4 \
    --woq_group_size 128 \
    --woq_scheme asym  \
    --woq_algo RTN \
    --woq_enable_mse_search \
    --output_dir "saved_results"
    --accuracy
```
### OPT-125m
#### Quantization
```bash
# GPTQ
python run_clm_no_trainer.py \
    --model facebook/opt-125m \
    --quantize \
    --approach weight_only \
    --woq_bits 4 \
    --woq_group_size 128 \
    --woq_scheme asym  \
    --woq_algo GPTQ \
    --woq_enable_mse_search \
    --output_dir "saved_results"
    --accuracy
```
**Notes**: Weight-only quantization based on fake quantization is previewly supported and supports RTN, GPTQ[1], AWQ[2], TEQ algorithms. For more details, please refer to [link](https://github.com/intel/neural-compressor/blob/master/docs/source/quantization_weight_only.md)


[1]. Elias, Frantar, et al. "GPTQ: Accurate Post-training Compression for Generative Pretrained Transformers." arXiv preprint arXiv:2210.17323 (2023).
[2]. Lin, Ji, et al. "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." arXiv preprint arXiv:2306.00978 (2023).
