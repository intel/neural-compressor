Step-by-Step
============
This document describes the step-by-step instructions to run large language models (LLMs) using Static Quantization on 4th Gen Intel® Xeon® Scalable Processor (codenamed Sapphire Rapids) with PyTorch and Intel® Extension for PyTorch.

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
python run_clm_no_trainer.py \
    --model EleutherAI/gpt-j-6B \
    --quantize \
    --alpha 1.0 \
    --ipex \
    --output_dir "saved_results"
```

### OPT-125m

#### Quantization

```bash
python run_clm_no_trainer.py \
    --model facebook/opt-125m \
    --quantize \
    --alpha 0.5 \
    --ipex \
    --output_dir "saved_results"
```

### LLAMA2-7b/13b/70b
>Note: LLAMA requires IPEX requirements >= 2.1 to get better accuracy.
#### Quantization

```bash
python run_clm_no_trainer.py \
    --model meta-llama/Llama-2-7b-hf \
    --quantize \
    --alpha 0.8 \
    --ipex \
    --output_dir "saved_results"
```