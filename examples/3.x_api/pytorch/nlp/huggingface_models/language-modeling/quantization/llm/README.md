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
# "--woq_algo GPTQ" is used to enable GPTQ algorithms
# "--double_quant_type BNB" is used to enable double quant algorithms
python run_clm_no_trainer.py \
    --model EleutherAI/gpt-j-6B \
    --dataset NeelNanda/pile-10k \
    --quantize \
    --approach weight_only \
    --woq_algo GPTQ \
    --woq_bits 4 \
    --woq_scheme asym \
    --woq_group_size 128 \
    --gptq_pad_max_length 2048 \
    --gptq_use_max_length \
    --accuracy \
    --tasks "lambada_openai" \
    --double_quant_type "BNB"

# "--woq_algo RTN" is used to enable RTN algorithms
python run_clm_no_trainer.py \
    --model EleutherAI/gpt-j-6B \
    --dataset NeelNanda/pile-10k \
    --quantize \
    --approach weight_only \
    --woq_algo RTN \
    --woq_bits 4 \
    --woq_scheme asym \
    --woq_group_size 128 \
    --accuracy \
    --tasks "lambada_openai" \
    --double_quant_type "BNB"
```
**Notes**: Weight-only quantization based on fake quantization is previewly supported and supports RTN, GPTQ[1], AWQ[2], TEQ algorithms. For more details, please refer to [link](https://github.com/intel/neural-compressor/blob/master/docs/source/quantization_weight_only.md). Our GPTQ API support various CLMs including GPTJ, OPTs, Blooms, Llamas, Falcons, MPTs, ChatGLMs, etc. Simply replace the "--model" argument with other models to quantize different CLMs with GPTQ.


### OPT-125m

#### Quantization

```bash
# "--approach weight_only" is used to enable weight only quantization.
# "--woq_algo GPTQ" is used to enable GPTQ algorithms
# "--double_quant_type BNB" is used to enable double quant algorithms
python run_clm_no_trainer.py \
    --model facebook/opt-125m \
    --dataset NeelNanda/pile-10k \
    --quantize \
    --approach weight_only \
    --woq_algo GPTQ \
    --woq_bits 4 \
    --woq_scheme asym \
    --woq_group_size 128 \
    --gptq_pad_max_length 2048 \
    --gptq_use_max_length \
    --accuracy \
    --tasks "lambada_openai" \
    --double_quant_type "BNB"

# "--woq_algo RTN" is used to enable RTN algorithms
python run_clm_no_trainer.py \
    --model facebook/opt-125m \
    --dataset NeelNanda/pile-10k \
    --quantize \
    --approach weight_only \
    --woq_algo RTN \
    --woq_bits 4 \
    --woq_scheme asym \
    --woq_group_size 128 \
    --accuracy \
    --tasks "lambada_openai" \
    --double_quant_type "BNB"
```

### LLAMA2-7b/13b/30b
#### Quantization

```bash
# "--approach weight_only" is used to enable weight only quantization.
# "--double_quant_type BNB" is used to enable double quant algorithms
# "--woq_algo GPTQ" is used to enable GPTQ algorithms
python run_clm_no_trainer.py \
    --model meta-llama/Llama-2-7b-hf \
    --dataset NeelNanda/pile-10k \
    --quantize \
    --approach weight_only \
    --woq_algo GPTQ \
    --woq_bits 4 \
    --woq_scheme asym \
    --woq_group_size 128 \
    --gptq_pad_max_length 2048 \
    --gptq_use_max_length \
    --accuracy \
    --tasks "lambada_openai" \
    --double_quant_type "BNB"

# "--woq_algo RTN" is used to enable RTN algorithms
python run_clm_no_trainer.py \
    --model meta-llama/Llama-2-7b-hf \
    --dataset NeelNanda/pile-10k \
    --quantize \
    --approach weight_only \
    --woq_algo RTN \
    --woq_bits 4 \
    --woq_scheme asym \
    --woq_group_size 128 \
    --accuracy \
    --tasks "lambada_openai" \
    --double_quant_type "BNB"
```


[1]. Elias, Frantar, et al. "GPTQ: Accurate Post-training Compression for Generative Pretrained Transformers." arXiv preprint arXiv:2210.17323 (2023).
[2]. Lin, Ji, et al. "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." arXiv preprint arXiv:2306.00978 (2023).
