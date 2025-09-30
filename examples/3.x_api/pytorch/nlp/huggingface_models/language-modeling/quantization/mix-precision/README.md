# Run
 
In this example, you can verify the accuracy on HPU/CUDA device with emulation of MXFP4, MXFP8, NVFP4 and uNVFP4.

## Requirement

```bash
# neural-compressor-pt
pip install neural-compressor-pt==3.6
# auto-round
pip install auto-round==0.8.0
# other requirements
pip install -r requirements.txt
```
**Before neural-compressor v3.6 and auto-round v0.8.0 release, please install from source for the latest updates:**
```bash 
# neural-compressor-pt
INC_PT_ONLY=1 pip install git+https://github.com/intel/neural-compressor.git@v3.6rc
# auto-round
pip install git+https://github.com/intel/auto-round.git@v0.8.0rc
# other requirements
pip install -r requirements.txt
```

## Quantization

### Demo (`MXFP4`, `MXFP8`, `NVFP4`, `uNVFP4`)

```bash
python quantize.py  --model_name_or_path facebook/opt-125m --quantize --dtype MXFP4 --batch_size 8 --accuracy
```

### Mix-precision Quantization (`MXFP4 + MXFP8`)

```bash
# Llama 3.1 8B
python quantize.py  \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --quantize \
    --dtype MXFP4 \
    --use_recipe \
    --recipe_file recipes/Meta-Llama-3.1-8B-Instruct_7bits.json \
    --accuracy \
    --batch_size 32

# Llama 3.3 70B
deepspeed --include="localhost:4,5,6,7" --master_port=29500 python quantize.py  \
    --model_name_or_path meta-llama/Llama-3.3-70B-Instruct/ \
    --quantize \
    --dtype MXFP4 \
    --use_recipe \
    --recipe_file recipes/Meta-Llama-3.3-70B-Instruct_5bits.json \
    --accuracy \
    --batch_size 32
```

> Note: 
> 1. Quantization applies `--dtype` for all blocks in the model by removing `--use_recipe`.
> 2. Setting `--quant_lm_head` applies `--dtype` for the lm_head layer.
> 3. Setting `--iters 0` skips AutoRound tuning and uses RTN method.
> 4. The `deepspeed` usage provides quick accuracy verification.

## Inference usage

### NVFP4
NVFP4 is supported by vLLM already, the saved model in this example follows the `llm_compressor` format, please refer to the usage in the public vLLM document.

```bash
# Command to save model:
python quantize.py  --model_name_or_path facebook/opt-125m --quantize --dtype NVFP4 --batch_size 8 --save --save_path opt-125m-nvfp4 --save_format llm_compressor
```

### MXFP4 / MXFP8
MXFP4 and MXFP8 is enabled in a forked vLLM repo, usages as below:
```bash
# Install the forked vLLM
git clone -b cuda-mxfp8-moe --single-branch --quiet https://github.com/yiliu30/vllm-fork.git && cd vllm-fork
USE_CPP=0 VLLM_USE_PRECOMPILED=1 pip install -e . -vvv && cd -

# Command to save model:
python quantize.py  \
    --model_name_or_path meta-llama/Llama-3.3-70B-Instruct/ \
    --quantize \
    --iters 0 \
    --dtype MXFP4 \
    --save_path Llama-3.3-70B-Instruct-MXFP4 \
    --save \
    --save_format llm_compressor

# Command to inference with vLLM:
CUDA_VISIBLE_DEVICES=0,1 VLLM_USE_V1=0 VLLM_USE_MXFP4_CT_EMULATIONS=1 VLLM_LOGGING_LEVEL=DEBUG \
vllm serve Llama-3.3-70B-Instruct-MXFP4 --tensor-parallel-size=2 --port 7777 --host localhost --trust-remote-code --dtype bfloat16 --enforce-eager
export no_proxy="localhost, 127.0.0.1, ::1"
curl -X POST http://localhost:7777/v1/completions \
     -H "Content-Type: application/json" \
     -d '{
           "model": "/model_path/Llama-3.3-70B-Instruct-MXFP4",
           "prompt": "Solve the following math problem step by step: What is 25 + 37? Please answer directly with the result.",
           "max_tokens": 100,
           "temperature": 0.7,
           "top_p": 1.0
         }'
```
> Note: To inference with transformers, please save model with `--save_format auto_round` and try `python run_hf_inf.py ${model_name_or_path}`

### MXFP4 + MXFP8
Model with mixed precision is not supported in vLLM, but supported in transformers in `auto-round` format. 

```bash
# Command to save model:
python quantize.py  \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --quantize \
    --iters 0 \
    --dtype MXFP4 \
    --use_recipe \
    --recipe_file recipes/Meta-Llama-3.1-8B-Instruct_7bits.json \
    --save \
    --save_format auto_round \
    --save_path Llama-3.1-8B-Instruct-MXFP4-MXFP8-AR

# Command to inference with transformer:
python run_hf_inf.py Llama-3.1-8B-Instruct-MXFP4-MXFP8-AR
```
