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
CUDA_VISIBLE_DEVICES=1 python quantize.py  \
    --model_name_or_path facebook/opt-125m  \
    --quantize \
    --dtype MXFP4 \
    --enable_torch_compile \
    --low_gpu_mem_usage \
    --export_format auto_round \
    --export_path OPT-125M-MXFP4 \
    --accuracy \
    --tasks lambada_openai \
    --eval_batch_size 8  \
    --device_map 0
```

To verify the accuracy of the saved model:

```bash
# verify accuracy from export_path
CUDA_VISIBLE_DEVICES=2 python quantize.py \
    --model_name_or_path OPT-125M-MXFP4 \
    --accuracy \
    --tasks lambada_openai \
    --device_map 0  
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
CUDA_VISIBLE_DEVICES=1 python quantize.py \
    --model_name_or_path /models/Meta-Llama-3.1-8B-Instruct \
    --quantize \
    --dtype MXFP4 \
    --use_recipe \
    --recipe_file recipes/Meta-Llama-3.1-8B-Instruct_7bits.json \
    --enable_torch_compile \
    --low_gpu_mem_usage \
    --export_format auto_round \
    --export_path Llama-3.1-8B-MXFP4-MXFP8 \
    --device_map 0

# Llama 3.3 70B
CUDA_VISIBLE_DEVICES=1 python quantize.py  \
    --model_name_or_path meta-llama/Llama-3.3-70B-Instruct/ \
    --quantize \
    --dtype MXFP4 \
    --use_recipe \
    --recipe_file recipes/Meta-Llama-3.3-70B-Instruct_5bits.json \
    --enable_torch_compile \
    --low_gpu_mem_usage \
    --export_format auto_round \
    --export_path Llama-3.3-70B-MXFP4-MXFP8 \
    --device_map 0
```

Note: Please add more available cards by setting device_map if you got OOM issue.

#### Target_bits

To achieve optimal compression ratios in mixed-precision quantization, we provide the `--target_bits` argument for automated precision configuration.

- If you pass a single float number, it will automatically generate an optimal quantization recipe to achieve that target average bit-width.
- If you pass multiple float numbers, it will generate multiple recipes for different target bit-widths, allowing you to compare trade-offs between model size and accuracy.

Example usage:

```bash
CUDA_VISIBLE_DEVICES=1 python quantize.py  \
    --model_name_or_path facebook/opt-125m \
    --quantize \
    --dtype MXFP4 \
    --target_bits 6.5 7 7.3 \
    --tune_limit 100 \
    --enable_torch_compile \
    --low_gpu_mem_usage \
    --export_format auto_round \
    --export_path OPT-125m-MXFP4-MXFP8 \
    --accuracy \
    --tasks lambada_openai \
    --eval_batch_size 8 \
    --device_map 0
```


## Inference usage


### MXFP4 / MXFP8
MXFP4 and MXFP8 is enabled in a forked vLLM repo, usages as below:
```bash
# Install the forked vLLM
git clone https://github.com/yiliu30/vllm-fork.git
cd vllm-fork
git checkout fused-moe-ar
VLLM_USE_PRECOMPILED=1 pip install -e .
# Run accuracy evaluation
CUDA_VISIBLE_DEVICES=1 python lm_eval_launcher.py --enable-ar-ext --model vllm \
    --model_args pretrained=Qwen3-30B-A3B-MXFP4,tensor_parallel_size=1,data_parallel_size=1 \
    --tasks lambada_openai \
    --batch_size 8


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
