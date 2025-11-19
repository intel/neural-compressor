# Run
 
In this example, you can verify the accuracy on HPU/CUDA device with emulation of MXFP4, MXFP8, NVFP4 and uNVFP4.

## Requirement

```bash
# neural-compressor-pt
pip install neural-compressor-pt>=3.6
# auto-round
pip install auto-round>=0.8.0
# other requirements
pip install -r requirements.txt
```

## Quantization

### Demo (`MXFP4`, `MXFP8`, `NVFP4`, `uNVFP4`)

```bash
python quantize.py  \
    --model_name_or_path facebook/opt-125m \
    --quantize \
    --dtype MXFP4 \
    --enable_torch_compile \
    --low_gpu_mem_usage \
    --export_format auto_round \
    --export_path OPT-125m-MXFP4 \
    --accuracy \
    --eval_batch_size 8 \
```

Notes:
- Use `--export_format auto_round` for `MXFP4`, `MXFP8` data type
- Use `--export_format llm_compressor` for `NVFP4` data type
- Use `--export_format fake` for `uNVFP4` data type
- Use `--eval_parallelize` for model parallelism during accuracy evaluation.
- Setting `--quant_lm_head` applies `--dtype` for the lm_head layer.
- Setting `--iters 0` skips AutoRound tuning and uses RTN method.


### Mix-precision Quantization (`MXFP4 + MXFP8`)

```bash
# Llama 3.1 8B
python quantize.py  \
    --model_name_or_path /models/Meta-Llama-3.1-8B-Instruct \
    --quantize \
    --dtype MXFP4 \
    --use_recipe \
    --recipe_file recipes/Meta-Llama-3.1-8B-Instruct_7bits.json \
    --enable_torch_compile \
    --low_gpu_mem_usage \
    --export_format auto_round \
    --export_path Llama-3.1-8B-MXFP4-MXFP8

# Llama 3.3 70B
python quantize.py  \
    --model_name_or_path meta-llama/Llama-3.3-70B-Instruct/ \
    --quantize \
    --dtype MXFP4 \
    --use_recipe \
    --recipe_file recipes/Meta-Llama-3.3-70B-Instruct_5bits.json \
    --enable_torch_compile \
    --low_gpu_mem_usage \
    --export_format auto_round \
    --export_path Llama-3.3-70B-MXFP4-MXFP8
```

#### Target_bits

To achieve optimal compression ratios in mixed-precision quantization, we provide the `--target_bits` argument for automated precision configuration.

- If you pass a single float number, it will automatically generate an optimal quantization recipe to achieve that target average bit-width.
- If you pass multiple float numbers, it will generate multiple recipes for different target bit-widths, allowing you to compare trade-offs between model size and accuracy.

Example usage:

```bash
python quantize.py  \
    --model_name_or_path facebook/opt-125m \
    --quantize \
    --dtype MXFP4 \
    --target_bits 6.5 7 7.3 \
    --tune_limit 100 \
    --enable_torch_compile \
    --low_gpu_mem_usage \
    --export_format auto_round \
    --export_path OPT-125m-MXFP4 \
    --accuracy \
    --tasks mmlu \
    --eval_batch_size 8
```


## Inference usage


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
    --dtype MXFP4 \
    --use_recipe \
    --recipe_file recipes/Meta-Llama-3.1-8B-Instruct_7bits.json \
    --save \
    --save_format auto_round \
    --save_path Llama-3.1-8B-Instruct-MXFP4-MXFP8-AR \
    --enable_torch_compile

# Command to inference with transformer:
python run_hf_inf.py Llama-3.1-8B-Instruct-MXFP4-MXFP8-AR
```

### NVFP4
NVFP4 is supported by vLLM already, the saved model in this example follows the `llm_compressor` format.

```bash
lm_eval --model vllm \
    --model_args pretrained={nvfp4_model_path},tensor_parallel_size=1,data_parallel_size=1 \
    --tasks mmlu \
    --batch_size 4
```

### uNVFP4
uNVFP4 is saved in fake format which is actually high-precision. To verify accuracy, setting `--accuracy --tasks mmlu` in command.
