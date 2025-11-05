Step-by-Step
============
This document describes the step-by-step instructions to run large language models (LLMs) on 4th Gen Intel® Xeon® Scalable Processor (codenamed Sapphire Rapids) with PyTorch 2 Export Quantization.

Currently, users can use `run_clm_no_trainer.py` to quantize the `OPT` series models and validate the last word prediction accuracy with [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness.git). We will add more models in the near future.

# Prerequisite
## 1. Create Environment
```
# Installation
pip install -r requirements.txt
```

# Run

Here is how to run the scripts:

**Causal Language Modeling (CLM)**

`run_clm_no_trainer.py` quantizes the large language models using the dataset [NeelNanda/pile-10k](https://huggingface.co/datasets/NeelNanda/pile-10k) validates `lambada_openai`, `piqa`, `winogrande`, `hellaswag` and other datasets accuracy provided by lm_eval, an example command is as follows.
### OPT-125m

#### Quantization

```bash
python run_clm_no_trainer.py --model facebook/opt-125m --quantize --output_dir qmodel_save_path
```

#### Accuracy
```bash
# Measure the accuracy of the floating model
python run_clm_no_trainer.py --model facebook/opt-125m --accuracy  --tasks lambada_openai

# Measure the accuracy of the quantized model
python run_clm_no_trainer.py --model facebook/opt-125m --accuracy  --tasks lambada_openai --int8 --output_dir qmodel_save_path 
```

#### Performance
```bash
# Measure the performance of the floating model
python run_clm_no_trainer.py --model facebook/opt-125m --performance

# Measure the performance of the quantized model
python run_clm_no_trainer.py --model facebook/opt-125m --performance --int8 --output_dir qmodel_save_path 
```
