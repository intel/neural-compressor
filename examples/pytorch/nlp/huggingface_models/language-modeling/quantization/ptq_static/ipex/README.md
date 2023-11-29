Step-by-Step
============
This document describes the step-by-step instructions to run large language models (LLMs) on 4th Gen Intel® Xeon® Scalable Processor (codenamed Sapphire Rapids) with PyTorch and Intel® Extension for PyTorch.

The script `run_clm_no_trainer.py` supports `GPTJ`, `OPT`, `LLaMA`, `BLOOM`, `MPT` and `Falcon` quantization and validates last word prediction accuracy with [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness.git) now, and we are adding more models.

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
# "--sq" is used to enable smooth quant
# "--int8_bf16_mixed" is used to enable int8-bf16 mixed mode for platform that natively supports bf16
# "--peft_model_id" is used to loaded PEFT weights from peft_model_id
python run_clm_no_trainer.py \
    --model EleutherAI/gpt-j-6B \
    --quantize \
    --sq \
    --alpha 1.0 \
    --output_dir "saved_results" \
    --ipex \
    --peft_model_id "peft_model_id"
```

```bash
# "--approach weight_only" is used to enable weight only quantization.
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
```
**Notes**: Weight-only quantization based on fake quantization is previewly supported and supports RTN, GPTQ[1], AWQ[2], TEQ algorithms. For more details, please refer to [link](https://github.com/intel/neural-compressor/blob/master/docs/source/quantization_weight_only.md)


```bash
python run_clm_no_trainer.py \
    --model EleutherAI/gpt-j-6B \
    --woq_algo GPTQ \
    --woq_bits 4 \
    --quantize \
    --pad_max_length 2048 \
    --gptq_pad_max_length 2048 \
    --gptq_use_max_length \
    --approach weight_only \
    --output_dir "test_models" \
```


#### Accuracy with lm_eval
```bash
# FP32 Accuracy
python run_clm_no_trainer.py \
    --model EleutherAI/gpt-j-6B \
    --accuracy \
    --batch_size 112 \
    --tasks "lambada_openai" "lambada_standard"\
    --int8 \
    --ipex \
    --output_dir "saved_results"  # load int8 model
# to validate FP32 model, please remove "--int8" and "--output_dir".
```
### OPT-1.3b/2.7b/6.7b

#### Quantization

```bash
# "--sq" is used to enable smooth quant
# "--int8_bf16_mixed" is used to enable int8-bf16 mixed mode for platform that natively supports bf16
# "--peft_model_id" is used to loaded PEFT weights from peft_model_id
python run_clm_no_trainer.py \
    --model facebook/opt-2.7b \
    --quantize \
    --sq \
    --alpha 0.5 \
    --ipex \
    --output_dir "saved_results" \
    --int8_bf16_mixed \
    --peft_model_id "peft_model_id"
```

#### Accuracy with lm_eval
```bash
python run_clm_no_trainer.py \
    --model facebook/opt-2.7b \
    --accuracy \
    --batch_size 112 \
    --tasks "lambada_openai" "lambada_standard" \
    --int8 \
    --ipex \
    --output_dir "saved_results"  # load int8 model
# to validate FP32 model, please remove "--int8" and "--output_dir".
```
### LLAMA-7b/13b/30b
>Note: LLAMA requires IPEX requirements >= 2.1 to get better accuracy, please source install from [intel_extension_for_pytorch](https://github.com/intel/intel-extension-for-pytorch.git).
#### Quantization

```bash
# "--sq" is used to enable smooth quant
# "--int8_bf16_mixed" is used to enable int8-bf16 mixed mode for platform that natively supports bf16
# "--peft_model_id" is used to loaded PEFT weights from peft_model_id
python run_clm_no_trainer.py \
    --model decapoda-research/llama-7b-hf \
    --quantize \
    --sq \
    --alpha 0.8 \
    --ipex \
    --output_dir "saved_results" \
    --int8_bf16_mixed \
    --peft_model_id "peft_model_id"
```

#### Accuracy with lm_eval
```bash
python run_clm_no_trainer.py \
    --model decapoda-research/llama-7b-hf \
    --accuracy \
    --batch_size 112 \
    --tasks  "lambada_openai" "lambada_standard" \
    --int8 \
    --ipex \
    --output_dir "saved_results"  # load int8 model
# to validate FP32 model, please remove "--int8" and "--output_dir".
```

### MPT-7b-chat
#### Quantization
```bash
# "--sq" is used to enable smooth quant
# "--int8_bf16_mixed" is used to enable int8-bf16 mixed mode for platform that natively supports bf16
python run_clm_no_trainer.py \
    --model mosaicml/mpt-7b-chat \
    --quantize \
    --sq \
    --alpha 0.85 \
    --ipex \
    --output_dir "saved_results"
```

#### Accuracy with lm_eval
```bash
python run_clm_no_trainer.py \
    --model mosaicml/mpt-7b-chat \
    --accuracy \
    --batch_size 112 \
    --tasks  "lambada_openai" \
    --int8 \
    --ipex \
    --output_dir "saved_results"  # load int8 model
# to validate FP32 model, please remove "--int8" and "--output_dir".
```
### Falcon-7b-instruct
#### Quantization
```bash
# "--sq" is used to enable smooth quant
# "--int8_bf16_mixed" is used to enable int8-bf16 mixed mode for platform that natively supports bf16
python run_clm_no_trainer.py \
    --model tiiuae/falcon-7b-instruct \
    --quantize \
    --sq \
    --alpha 0.7 \
    --output_dir "saved_results"
```

#### Accuracy with lm_eval
```bash
python run_clm_no_trainer.py \
    --model tiiuae/falcon-7b-instruct \
    --accuracy \
    --batch_size 112 \
    --tasks  "lambada_openai" \
    --int8 \
    --ipex \
    --output_dir "saved_results"  # load int8 model
# to validate FP32 model, please remove "--int8" and "--output_dir".
```

[1]. Elias, Frantar, et al. "GPTQ: Accurate Post-training Compression for Generative Pretrained Transformers." arXiv preprint arXiv:2210.17323 (2023).
[2]. Lin, Ji, et al. "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." arXiv preprint arXiv:2306.00978 (2023).
