# Step-by-Step recipes for LLM quantization

This document describes the step-by-step instructions to run large language models (LLMs) on 4th Gen Intel® Xeon® Scalable Processor (codenamed Sapphire Rapids) with [PyTorch](https://pytorch.org/) and [Intel® Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch).

The scripts [run_generation_sq.py](./run_generation_sq.py) and [run_generation_cpu_woq.py](./run_generation_cpu_woq.py) provide two quantization approaches respectively (SmoothQuant, Weight-Only Quantization) based on [Intel® Neural Compressor](https://github.com/intel/neural-compressor) and return last word prediction accuracy by [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/master).

# Validated Models

|                             Model Name                              |
| :-----------------------------------------------------------------: |
|             [EleutherAI/gpt-j-6b](#eleutheraigpt-j-6b)              |
|                [facebook/opt-1.3b](#facebookopt-13b)                |
|                [facebook/opt-30b](#facebookopt-30b)                 |
|        [meta-llama/Llama-2-7b-hf](#meta-llamallama-2-7b-hf)         |
|       [meta-llama/Llama-2-13b-hf](#meta-llamallama-2-13b-hf)        |
|       [meta-llama/Llama-2-70b-hf](#meta-llamallama-2-70b-hf)        |
|               [tiiuae/falcon-40b](#tiiuaefalcon-40b)                |
|                [tiiuae/falcon-7b](#tiiuaefalcon-7b)                 |
|  [baichuan-inc/Baichuan2-7B-Chat](#baichuan-inc/Baichuan2-7B-Chat)  |
| [baichuan-inc/Baichuan2-13B-Chat](#baichuan-inc/Baichuan2-13B-Chat) |
|  [baichuan-inc/Baichuan-13B-Chat](#baichuan-inc/Baichuan-13B-Chat)  |
|               [THUDM/chatglm2-6b](#THUDM/chatglm2-6b)               |
|               [THUDM/chatglm3-6b](#THUDM/chatglm3-6b)               |
|            [bigscience/bloom-1b7](#bigscience/bloom-1b7)            |
|         [EleutherAI/gpt-neox-20b](#EleutherAI/gpt-neox-20b)         |
|       [mistralai/Mistral-7B-v0.1](#mistralai/Mistral-7B-v0.1)       |
|         [databricks/dolly-v2-12b](#databricks/dolly-v2-12b)         |

# Prerequisite

```bash
# Installation
git clone https://github.com/intel/intel-extension-for-transformers.git

# install ITREX
cd intel-extension-for-transformers
pip install -r requirements.txt
pip install -v .

# install requirements
cd examples/huggingface/pytorch/text-generation/quantization
pip install -r requirements.txt
pip install neural-compressor==3.0
pip install torch==2.3.0+cpu --index-url https://download.pytorch.org/whl/cpu
# 4.38.1 is only limited by smoothquant
pip install transformers==4.38.2  # 4.42.4 for mistralai/Mistral-7B-v0.1
# ipex is only necessary for smoothquant
pip install intel-extension-for-pytorch==2.3.0
```

# Run Quantization and evaluate INT8 accuracy

## EleutherAI/gpt-j-6b

### SmoothQuant

```bash
python run_generation_sq.py \
    --model EleutherAI/gpt-j-6b \
    --output_dir ./saved_results \
    --tasks lambada_openai \
    --sq \
    --accuracy \
    --eval_batch_size 1 \
    --alpha 0.85
```

### Weight-Only Quantization

```bash
# int8 RTN
python run_generation_cpu_woq.py \
    --model EleutherAI/gpt-j-6b \
    --output_dir ./saved_results \
    --woq \
    --accuracy

# int4 GPTQ
python run_generation_cpu_woq.py \
    --model EleutherAI/gpt-j-6b \
    --output_dir ./saved_results \
    --woq \
    --woq_algo GPTQ \
    --bits 4 \
    --weight_dtype int4 \
    --desc_act \
    --seq_len 2048 \
    --scheme sym \
    --group_size 32 \
    --accuracy


# int4 AutoRound
python run_generation_cpu_woq.py \
    --model EleutherAI/gpt-j-6b \
    --output_dir ./saved_results \
    --woq \
    --woq_algo AutoRound \
    --bits 4 \
    --weight_dtype int4 \
    --autoround_iters 200 \
    --scheme asym \
    --group_size 128 \
    --accuracy
```

## facebook/opt-1.3b

### SmoothQuant

```bash
python run_generation_sq.py \
    --model facebook/opt-1.3b \
    --output_dir ./saved_results \
    --tasks lambada_openai \
    --sq \
    --accuracy \
    --eval_batch_size 1 \
    --alpha 0.9
```

### Weight-Only Quantization

```bash
# int8 RTN
python run_generation_cpu_woq.py \
    --model facebook/opt-1.3b \
    --output_dir ./saved_results \
    --woq \
    --accuracy

# int4 GPTQ
python run_generation_cpu_woq.py \
    --model facebook/opt-1.3b \
    --output_dir ./saved_results \
    --woq \
    --woq_algo GPTQ \
    --bits 4 \
    --weight_dtype int4 \
    --desc_act \
    --seq_len 2048 \
    --scheme sym \
    --group_size 128 \
    --accuracy

# int4 AutoRound
python run_generation_cpu_woq.py \
    --model facebook/opt-1.3b \
    --output_dir ./saved_results \
    --woq \
    --woq_algo AutoRound \
    --bits 4 \
    --weight_dtype int4 \
    --autoround_iters 200 \
    --scheme asym \
    --group_size 128 \
    --accuracy
```

## facebook/opt-30b

### SmoothQuant

```bash
python run_generation_sq.py \
    --model facebook/opt-30b \
    --output_dir ./saved_results \
    --trust_remote_code \
    --tasks lambada_openai \
    --sq \
    --accuracy \
    --eval_batch_size 1 \
    --alpha 0.5
```

### Weight-Only Quantization

```bash
# int8 RTN
python run_generation_cpu_woq.py \
    --model facebook/opt-30b \
    --output_dir ./saved_results \
    --woq \
    --accuracy

# int4 GPTQ
python run_generation_cpu_woq.py \
    --model facebook/opt-30b \
    --output_dir ./saved_results \
    --woq \
    --woq_algo GPTQ \
    --bits 4 \
    --weight_dtype int4 \
    --desc_act \
    --seq_len 2048 \
    --scheme sym \
    --group_size 32 \
    --accuracy

# int4 AutoRound
python run_generation_cpu_woq.py \
    --model facebook/opt-30b \
    --output_dir ./saved_results \
    --woq \
    --woq_algo AutoRound \
    --bits 4 \
    --weight_dtype int4 \
    --autoround_iters 200 \
    --scheme asym \
    --group_size 128 \
    --accuracy
```

## meta-llama/Llama-2-7b-hf

### SmoothQuant

```bash
python run_generation_sq.py \
    --model meta-llama/Llama-2-7b-hf \
    --output_dir ./saved_results \
    --tasks lambada_openai \
    --sq \
    --accuracy \
    --eval_batch_size 1 \
    --init_alpha 0.8 \
    --alpha_min 0.79 \
    --alpha_max 0.99 \
    --alpha_step 0.01 \
    --shared_criterion mean \
    --seq_len 2048 \
    --shuffle \
    --n_samples 512 \
    --alpha auto
```

### Weight-Only Quantization

```bash
# int8 RTN
python run_generation_cpu_woq.py \
    --model meta-llama/Llama-2-7b-hf \
    --output_dir ./saved_results \
    --woq \
    --accuracy

# int4 GPTQ
python run_generation_cpu_woq.py \
    --model meta-llama/Llama-2-7b-hf \
    --output_dir ./saved_results \
    --woq \
    --woq_algo GPTQ \
    --bits 4 \
    --weight_dtype int4 \
    --desc_act \
    --seq_len 2048 \
    --scheme sym \
    --group_size 32 \
    --accuracy

# int4 AutoRound
python run_generation_cpu_woq.py \
    --model meta-llama/Llama-2-7b-hf \
    --output_dir ./saved_results \
    --woq \
    --woq_algo AutoRound \
    --bits 4 \
    --weight_dtype int4 \
    --autoround_iters 200 \
    --scheme asym \
    --group_size 128 \
    --accuracy
```

## meta-llama/Llama-2-13b-hf

### SmoothQuant

```bash
python run_generation_sq.py \
    --model meta-llama/Llama-2-13b-hf \
    --output_dir ./saved_results \
    --tasks lambada_openai \
    --sq \
    --accuracy \
    --eval_batch_size 1 \
    --seq_len 1024 \
    --init_alpha 0.8 \
    --alpha_min 0.75 \
    --alpha_max 0.99 \
    --alpha_step 0.01 \
    --shared_criterion max \
    --padding \
    --n_samples 512 \
    --alpha auto
```

### Weight-Only Quantization

```bash
# int8 RTN
python run_generation_cpu_woq.py \
    --model meta-llama/Llama-2-13b-hf \
    --output_dir ./saved_results \
    --woq \
    --accuracy

# int4 GPTQ
python run_generation_cpu_woq.py \
    --model meta-llama/Llama-2-13b-hf \
    --output_dir ./saved_results \
    --woq \
    --woq_algo GPTQ \
    --bits 4 \
    --weight_dtype int4 \
    --desc_act \
    --seq_len 2048 \
    --scheme sym \
    --group_size 32 \
    --accuracy

# int4 AutoRound
python run_generation_cpu_woq.py \
    --model meta-llama/Llama-2-13b-hf \
    --output_dir ./saved_results \
    --woq \
    --woq_algo AutoRound \
    --bits 4 \
    --weight_dtype int4 \
    --autoround_iters 200 \
    --scheme asym \
    --group_size 128 \
    --accuracy
```

## meta-llama/Llama-2-70b-hf

### SmoothQuant

```bash
python run_generation_sq.py \
    --model meta-llama/Llama-2-70b-hf \
    --output_dir ./saved_results \
    --tasks lambada_openai \
    --sq \
    --accuracy \
    --eval_batch_size 1 \
    --alpha 0.8 \
    --n_samples 512
```

### Weight-Only Quantization

```bash
# int8 RTN
python run_generation_cpu_woq.py \
    --model meta-llama/Llama-2-70b-hf \
    --output_dir ./saved_results \
    --woq \
    --accuracy

# int4 GPTQ
python run_generation_cpu_woq.py \
    --model meta-llama/Llama-2-70b-hf \
    --output_dir ./saved_results \
    --woq \
    --woq_algo GPTQ \
    --bits 4 \
    --weight_dtype int4 \
    --desc_act \
    --seq_len 2048 \
    --scheme sym \
    --group_size 32 \
    --accuracy

# int4 AutoRound
python run_generation_cpu_woq.py \
    --model meta-llama/Llama-2-70b-hf \
    --output_dir ./saved_results \
    --woq \
    --woq_algo AutoRound \
    --bits 4 \
    --weight_dtype int4 \
    --autoround_iters 200 \
    --scheme asym \
    --group_size 128 \
    --accuracy
```

## tiiuae/falcon-40b

### SmoothQuant

```bash
python run_generation_sq.py \
    --model tiiuae/falcon-40b \
    --output_dir ./saved_results \
    --trust_remote_code \
    --tasks lambada_openai \
    --sq \
    --accuracy \
    --eval_batch_size 1 \
    --alpha 0.9
```

### Weight-Only Quantization

```bash
# int8 RTN
python run_generation_cpu_woq.py \
    --model tiiuae/falcon-40b \
    --output_dir ./saved_results \
    --woq \
    --accuracy

# int4 GPTQ
python run_generation_cpu_woq.py \
    --model tiiuae/falcon-40b \
    --output_dir ./saved_results \
    --woq \
    --woq_algo GPTQ \
    --bits 4 \
    --weight_dtype int4 \
    --desc_act \
    --seq_len 2048 \
    --scheme sym \
    --group_size 32 \
    --accuracy

# int4 AutoRound
python run_generation_cpu_woq.py \
    --model tiiuae/falcon-40b \
    --output_dir ./saved_results \
    --woq \
    --woq_algo AutoRound \
    --bits 4 \
    --weight_dtype int4 \
    --autoround_iters 200 \
    --scheme asym \
    --group_size 128 \
    --accuracy
```

## tiiuae/falcon-7b

### SmoothQuant

```bash
python run_generation_sq.py \
    --model tiiuae/falcon-7b \
    --output_dir ./saved_results \
    --trust_remote_code \
    --tasks lambada_openai \
    --sq --accuracy \
    --eval_batch_size 1 \
    --alpha 0.95
```

### Weight-Only Quantization

```bash
# int8 RTN
python run_generation_cpu_woq.py \
    --model tiiuae/falcon-7b \
    --output_dir ./saved_results \
    --woq \
    --accuracy

# int4 GPTQ
python run_generation_cpu_woq.py \
    --model  tiiuae/falcon-7b \
    --output_dir ./saved_results \
    --woq \
    --woq_algo GPTQ \
    --bits 4 \
    --weight_dtype int4 \
    --seq_len 2048 \
    --scheme sym \
    --group_size 32 \
    --accuracy

# int4 AutoRound
python run_generation_cpu_woq.py \
    --model tiiuae/falcon-7b \
    --output_dir ./saved_results \
    --woq \
    --woq_algo AutoRound \
    --bits 4 \
    --weight_dtype int4 \
    --autoround_iters 200 \
    --scheme asym \
    --group_size 128 \
    --accuracy
```

## baichuan-inc/Baichuan2-7B-Chat

### SmoothQuant

```bash
python run_generation_sq.py \
    --model baichuan-inc/Baichuan2-7B-Chat \
    --output_dir ./saved_results \
    --trust_remote_code \
    --tasks lambada_openai \
    --sq \
    --accuracy \
    --eval_batch_size 1 \
    --alpha 0.95
```

### Weight-Only Quantization

```bash
# int8 RTN
python run_generation_cpu_woq.py \
    --model baichuan-inc/Baichuan2-7B-Chat \
    --output_dir ./saved_results \
    --woq \
    --accuracy

# int4 GPTQ
python run_generation_cpu_woq.py \
    --model baichuan-inc/Baichuan2-7B-Chat \
    --output_dir ./saved_results \
    --woq \
    --woq_algo GPTQ \
    --bits 4 \
    --weight_dtype int4 \
    --desc_act \
    --seq_len 2048 \
    --scheme sym \
    --group_size 32 \
    --accuracy

# int4 AutoRound
python run_generation_cpu_woq.py \
    --model baichuan-inc/Baichuan2-7B-Chat \
    --output_dir ./saved_results \
    --woq \
    --woq_algo AutoRound \
    --bits 4 \
    --weight_dtype int4 \
    --autoround_iters 200 \
    --scheme asym \
    --group_size 128 \
    --accuracy
```

## baichuan-inc/Baichuan2-13B-Chat

### SmoothQuant

```bash
python run_generation_sq.py \
    --model baichuan-inc/Baichuan2-13B-Chat \
    --output_dir ./saved_results \
    --trust_remote_code \
    --tasks lambada_openai \
    --sq \
    --accuracy \
    --eval_batch_size 1 \
    --alpha 0.65
```

### Weight-Only Quantization

```bash
# int8 RTN
python run_generation_cpu_woq.py \
    --model baichuan-inc/Baichuan2-13B-Chat \
    --output_dir ./saved_results \
    --woq \
    --accuracy

# int4 GPTQ
python run_generation_cpu_woq.py \
    --model baichuan-inc/Baichuan2-13B-Chats \
    --output_dir ./saved_results \
    --woq \
    --woq_algo GPTQ \
    --bits 4 \
    --weight_dtype int4 \
    --desc_act \
    --seq_len 2048 \
    --scheme sym \
    --group_size 32 \
    --accuracy

# int4 AutoRound
python run_generation_cpu_woq.py \
    --model baichuan-inc/Baichuan2-13B-Chat \
    --output_dir ./saved_results \
    --woq \
    --woq_algo AutoRound \
    --bits 4 \
    --weight_dtype int4 \
    --autoround_iters 200 \
    --scheme asym \
    --group_size 128 \
    --accuracy
```

## baichuan-inc/Baichuan-13B-Chat

### SmoothQuant

```bash
python run_generation_sq.py \
    --model baichuan-inc/Baichuan-13B-Chat \
    --output_dir ./saved_results \
    --trust_remote_code \
    --tasks lambada_openai \
    --sq --accuracy \
    --eval_batch_size 1 \
    --alpha 0.5
```

### Weight-Only Quantization

```bash
# int8 RTN
python run_generation_cpu_woq.py \
    --model baichuan-inc/Baichuan-13B-Chat \
    --output_dir ./saved_results \
    --woq \
    --accuracy

# int4 GPTQ

python run_generation_cpu_woq.py \
    --model baichuan-inc/Baichuan-13B-Chat \
    --output_dir ./saved_results \
    --woq \
    --woq_algo GPTQ \
    --bits 4 \
    --weight_dtype int4 \
    --desc_act \
    --seq_len 2048 \
    --scheme sym \
    --group_size 32 \
    --accuracy

# int4 AutoRound
python run_generation_cpu_woq.py \
    --model baichuan-inc/Baichuan-13B-Chat \
    --output_dir ./saved_results \
    --woq \
    --woq_algo AutoRound \
    --bits 4 \
    --weight_dtype int4 \
    --autoround_iters 200 \
    --scheme asym \
    --group_size 128 \
    --accuracy
```

## THUDM/chatglm2-6b

### SmoothQuant

```bash
python run_generation_sq.py \
    --model THUDM/chatglm2-6b \
    --output_dir ./saved_results \
    --trust_remote_code \
    --tasks lambada_openai \
    --sq \
    --accuracy \
    --eval_batch_size 1 \
    --init_alpha 0.7 \
    --alpha_min 0.55 \
    --alpha_max 0.8 \
    --alpha_step 0.01 \
    --shared_criterion mean \
    --n_samples 512
```

### Weight-Only Quantization

```bash
# int8 RTN
python run_generation_cpu_woq.py \
    --model THUDM/chatglm2-6b \
    --output_dir ./saved_results \
    --woq \
    --accuracy

# int4 GPTQ
python run_generation_cpu_woq.py \
    --model THUDM/chatglm2-6b \
    --output_dir ./saved_results \
    --woq \
    --woq_algo GPTQ \
    --bits 4 \
    --weight_dtype int4 \
    --seq_len 2048 \
    --scheme asym \
    --group_size 32 \
    --accuracy

# int4 AutoRound
python run_generation_cpu_woq.py \
    --model THUDM/chatglm2-6b \
    --output_dir ./saved_results \
    --woq \
    --woq_algo AutoRound \
    --bits 4 \
    --weight_dtype int4 \
    --autoround_iters 200 \
    --scheme asym \
    --group_size 128 \
    --accuracy
```

## THUDM/chatglm3-6b

### SmoothQuant

```bash
python run_generation_sq.py \
    --model THUDM/chatglm3-6b \
    --output_dir ./saved_results \
    --trust_remote_code \
    --tasks lambada_openai \
    --sq \
    --accuracy \
    --eval_batch_size 1 \
    --init_alpha 0.85 \
    --alpha_min 0.79 \
    --alpha_max 0.88 \
    --alpha_step 0.01 \
    --shared_criterion mean
```

### Weight-Only Quantization

```bash
# int8 RTN
python run_generation_cpu_woq.py \
    --model THUDM/chatglm3-6b \
    --output_dir ./saved_results \
    --woq \
    --accuracy

# int4 GPTQ
python run_generation_cpu_woq.py \
    --model THUDM/chatglm3-6b \
    --output_dir ./saved_results \
    --woq \
    --woq_algo GPTQ \
    --bits 4 \
    --weight_dtype int4 \
    --seq_len 2048 \
    --scheme sym \
    --group_size 32 \
    --n_samples 256 \
    --accuracy

# int4 AutoRound
python run_generation_cpu_woq.py \
    --model THUDM/chatglm3-6b \
    --output_dir ./saved_results \
    --woq \
    --woq_algo AutoRound \
    --bits 4 \
    --weight_dtype int4 \
    --autoround_iters 200 \
    --scheme asym \
    --group_size 128 \
    --accuracy
```

## bigscience/bloom-1b7

### SmoothQuant

```bash
python run_generation_sq.py \
    --model bigscience/bloom-1b7 \
    --output_dir ./saved_results \
    --tasks lambada_openai \
    --sq \
    --accuracy \
    --eval_batch_size 1 \
    --alpha 0.6
```

### Weight-Only Quantization

```bash
# int8 RTN
python run_generation_cpu_woq.py \
    --model bigscience/bloom-1b7 \
    --output_dir ./saved_results \
    --woq \
    --accuracy

# int4 GPTQ
python run_generation_cpu_woq.py \
    --model bigscience/bloom-1b7 \
    --output_dir ./saved_results \
    --woq \
    --woq_algo GPTQ \
    --bits 4 \
    --weight_dtype int4 \
    --desc_act \
    --seq_len 2048 \
    --scheme sym \
    --group_size 32 \
    --accuracy

# int4 AutoRound
python run_generation_cpu_woq.py \
    --model bigscience/bloom-1b7 \
    --output_dir ./saved_results \
    --woq \
    --woq_algo AutoRound \
    --bits 4 \
    --weight_dtype int4 \
    --autoround_iters 200 \
    --scheme asym \
    --group_size 128 \
    --accuracy
```

## EleutherAI/gpt-neox-20b

### SmoothQuant

```bash
python run_generation_sq.py \
    --model EleutherAI/gpt-neox-20b \
    --output_dir ./saved_results \
    --tasks lambada_openai \
    --sq \
    --accuracy \
    --eval_batch_size 1 \
    --alpha 0.7
```

### Weight-Only Quantization

```bash
# int8 RTN
python run_generation_cpu_woq.py \
    --model EleutherAI/gpt-neox-20b \
    --output_dir ./saved_results \
    --woq \
    --accuracy

# int4 GPTQ
python run_generation_cpu_woq.py \
    --model EleutherAI/gpt-neox-20b \
    --output_dir ./saved_results \
    --woq \
    --woq_algo GPTQ \
    --bits 4 \
    --weight_dtype int4 \
    --seq_len 2048 \
    --scheme asym \
    --group_size 32 \
    --accuracy

# int4 AutoRound
python run_generation_cpu_woq.py \
    --model EleutherAI/gpt-neox-20b \
    --output_dir ./saved_results \
    --woq \
    --woq_algo AutoRound \
    --bits 4 \
    --weight_dtype int4 \
    --autoround_iters 200 \
    --scheme asym \
    --group_size 128 \
    --accuracy
```

## mistralai/Mistral-7B-v0.1

### SmoothQuant

```bash
python run_generation_sq.py \
    --model mistralai/Mistral-7B-v0.1 \
    --output_dir ./saved_results \
    --trust_remote_code \
    --tasks lambada_openai \
    --sq \
    --accuracy \
    --eval_batch_size 1 \
    --alpha 0.75
```

### Weight-Only Quantization

```bash
# int8 RTN
python run_generation_cpu_woq.py \
    --model mistralai/Mistral-7B-v0.1 \
    --output_dir ./saved_results \
    --woq \
    --accuracy

# int4 GPTQ
python run_generation_cpu_woq.py \
    --model mistralai/Mistral-7B-v0.1 \
    --output_dir ./saved_results \
    --woq \
    --woq_algo GPTQ \
    --bits 4 \
    --weight_dtype int4 \
    --scheme asym \
    --group_size 128 \
    --use_mse_search \
    --n_samples 128
    --accuracy

# int4 AutoRound
python run_generation_cpu_woq.py \
    --model mistralai/Mistral-7B-v0.1 \
    --output_dir ./saved_results \
    --woq \
    --woq_algo AutoRound \
    --bits 4 \
    --weight_dtype int4 \
    --autoround_iters 200 \
    --scheme asym \
    --group_size 128 \
    --accuracy
```

## databricks/dolly-v2-12b

### SmoothQuant

```bash
python run_generation_sq.py \
    --model databricks/dolly-v2-12b \
    --output_dir ./saved_results \
    --tasks lambada_openai \
    --sq \
    --accuracy \
    --eval_batch_size 1 \
    --alpha 0.75
```

### Weight-Only Quantization

```bash
# int8 RTN
python run_generation_cpu_woq.py \
    --model databricks/dolly-v2-12b \
    --output_dir ./saved_results \
    --woq \
    --accuracy

# int4 GPTQ
python run_generation_cpu_woq.py \
    --model databricks/dolly-v2-12b \
    --output_dir ./saved_results \
    --woq \
    --woq_algo GPTQ \
    --bits 4 \
    --weight_dtype int4 \
    --seq_len 2048 \
    --scheme sym \
    --group_size 128 \
    --accuracy

# int4 AutoRound
python run_generation_cpu_woq.py \
    --model databricks/dolly-v2-12b \
    --output_dir ./saved_results \
    --woq \
    --woq_algo AutoRound \
    --bits 4 \
    --weight_dtype int4 \
    --autoround_iters 200 \
    --scheme asym \
    --group_size 128 \
    --accuracy
```
