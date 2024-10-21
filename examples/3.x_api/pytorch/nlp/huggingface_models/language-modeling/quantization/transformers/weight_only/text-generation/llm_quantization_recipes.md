# Step-by-Step recipes for LLM quantization

This document describes the step-by-step instructions to run large language models (LLMs) on 5th Gen Intel® Xeon® Scalable Processor (codenamed Emerald Rapids) with [PyTorch](https://pytorch.org/).

The scripts [run_generation_cpu_woq.py](./run_generation_cpu_woq.py) provide Weight-Only Quantization based on [Intel® Neural Compressor](https://github.com/intel/neural-compressor) and return last word prediction accuracy by [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/master).

# Validated Models

|                                  Model Name                                   |
| :---------------------------------------------------------------------------: |
|                  [EleutherAI/gpt-j-6b](#eleutheraigpt-j-6b)                   |
|                     [facebook/opt-30b](#facebookopt-30b)                      |
|             [meta-llama/Llama-2-7b-hf](#meta-llamallama-2-7b-hf)              |
|            [meta-llama/Llama-2-13b-hf](#meta-llamallama-2-13b-hf)             |
| [meta-llama/Meta-Llama-3.1-8B-Instruct](#meta-llamameta-llama-31-8b-instruct) |
|                    [tiiuae/falcon-40b](#tiiuaefalcon-40b)                     |
|                     [tiiuae/falcon-7b](#tiiuaefalcon-7b)                      |
|       [baichuan-inc/Baichuan2-7B-Chat](#baichuan-inc/Baichuan2-7B-Chat)       |
|      [baichuan-inc/Baichuan2-13B-Chat](#baichuan-inc/Baichuan2-13B-Chat)      |
|                    [THUDM/chatglm2-6b](#THUDM/chatglm2-6b)                    |
|                    [THUDM/chatglm3-6b](#THUDM/chatglm3-6b)                    |
|            [mistralai/Mistral-7B-v0.1](#mistralai/Mistral-7B-v0.1)            |
| [microsoft/Phi-3-medium-128k-instruct](#microsoft/Phi-3-medium-128k-instruct) |
|   [microsoft/Phi-3-mini-128k-instruct](#microsoft/Phi-3-mini-128k-instruct)   |
|               [Qwen/Qwen2-7B-Instruct](#Qwen/Qwen2-7B-Instruct)               |

# Prerequisite

```bash
# install requirements
cd examples/huggingface/pytorch/text-generation/quantization
pip install -r requirements.txt
pip install neural-compressor==3.1
pip install torch==2.4.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.43.0
```

# Run Quantization and evaluate INT8/INT4 accuracy

## EleutherAI/gpt-j-6b

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

## facebook/opt-30b

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

## meta-llama/Meta-Llama-3.1-8B-Instruct

### Weight-Only Quantization

```bash
# int8 RTN
python run_generation_cpu_woq.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --output_dir ./saved_results \
    --woq \
    --accuracy

# int4 GPTQ
python run_generation_cpu_woq.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --output_dir ./saved_results \
    --woq \
    --woq_algo GPTQ \
    --bits 4 \
    --weight_dtype int4 \
    --desc_act \
    --scheme asym \
    --group_size 32 \
    --accuracy

# int4 AutoRound
python run_generation_cpu_woq.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
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

## THUDM/chatglm2-6b

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

## mistralai/Mistral-7B-v0.1

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

## microsoft/Phi-3-medium-128k-instruct

### Weight-Only Quantization

```bash
# int8 RTN
python run_generation_cpu_woq.py \
    --model microsoft/Phi-3-medium-128k-instruct \
    --output_dir ./saved_results \
    --woq \
    --accuracy

# int4 GPTQ
python run_generation_cpu_woq.py \
    --model microsoft/Phi-3-medium-128k-instruct \
    --output_dir ./saved_results \
    --woq \
    --woq_algo GPTQ \
    --bits 4 \
    --weight_dtype int4 \
    --scheme sym \
    --group_size 32 \
    --accuracy

# int4 AutoRound
python run_generation_cpu_woq.py \
    --model microsoft/Phi-3-medium-128k-instruct \
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

## microsoft/Phi-3-mini-128k-instruct

```bash
# int8 RTN
python run_generation_cpu_woq.py \
    --model microsoft/Phi-3-mini-128k-instruct \
    --output_dir ./saved_results \
    --woq \
    --accuracy

# int4 GPTQ
python run_generation_cpu_woq.py \
    --model microsoft/Phi-3-mini-128k-instruct \
    --output_dir ./saved_results \
    --woq \
    --woq_algo GPTQ \
    --bits 4 \
    --weight_dtype int4 \
    --scheme asym \
    --group_size 32 \
    --accuracy

# int4 AutoRound
python run_generation_cpu_woq.py \
    --model microsoft/Phi-3-mini-128k-instruct \
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

## Qwen/Qwen2-7B-Instruct

```bash
# int8 RTN
python run_generation_cpu_woq.py \
    --model Qwen/Qwen2-7B-Instruct \
    --output_dir ./saved_results \
    --woq \
    --accuracy

# int4 GPTQ
python run_generation_cpu_woq.py \
    --model Qwen/Qwen2-7B-Instruct \
    --output_dir ./saved_results \
    --woq \
    --woq_algo GPTQ \
    --bits 4 \
    --weight_dtype int4 \
    --desc_act \
    --scheme sym \
    --group_size 32 \
    --accuracy

# int4 AutoRound
python run_generation_cpu_woq.py \
    --model Qwen/Qwen2-7B-Instruct \
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
