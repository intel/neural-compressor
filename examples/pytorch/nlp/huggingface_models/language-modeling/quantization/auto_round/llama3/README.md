# Step-by-step
 
In this example, you can verify the accuracy on HPU/CUDA device with emulation of MXFP4, MXFP8, NVFP4 and uNVFP4.

## Requirement

```bash
# neural-compressor-pt
pip install neural-compressor-pt
# auto-round
pip install auto-round
# other requirements
pip install -r requirements.txt
```

## Quantization

### Demo (`MXFP4`, `MXFP8`, `NVFP4`, `uNVFP4`)

```bash
CUDA_VISIBLE_DEVICES=0 python quantize.py  \
    --model_name_or_path facebook/opt-125m  \
    --quantize \
    --dtype MXFP8 \
    --enable_torch_compile \
    --low_gpu_mem_usage \
    --export_format auto_round \
    --export_path OPT-125M-MXFP8 \
    --accuracy \
    --tasks lambada_openai \
    --eval_batch_size 8
```

Notes:
- Use `--export_format auto_round` for `MXFP4`, `MXFP8` data type and do inference as below.
- Use `--export_format llm_compressor` for `NVFP4` data type since public vLLM supports it.
- Use `--export_format fake` for `uNVFP4` data type since it's not fully supported.
- Setting `--quant_lm_head` applies `--dtype` for the lm_head layer.
- Setting `--iters 0` skips AutoRound tuning and uses RTN method.
- Removing `--quantize` to evaluate the original model accuracy.


#### Target_bits

To achieve optimal compression ratios in mixed-precision quantization, we provide the `--target_bits` argument for automated precision configuration.

- If you pass a single float number, it will automatically generate an optimal quantization recipe to achieve that target average bit-width.
- If you pass multiple float numbers, it will generate multiple recipes for different target bit-widths, allowing you to compare trade-offs between model size and accuracy.

Example usage:

```bash
CUDA_VISIBLE_DEVICES=0 python quantize.py  \
    --model_name_or_path facebook/opt-125m \
    --quantize \
    --dtype MXFP4 \
    --target_bits 7.1 7.2 7.3 \
    --options "MXFP4" "MXFP8" \
    --shared_layer "k_proj" "v_proj" "q_proj" \
    --shared_layer "fc1" "fc2" \
    --enable_torch_compile \
    --low_gpu_mem_usage \
    --export_format auto_round \
    --export_path OPT-125m-MXFP4-MXFP8 \
    --tasks lambada_openai \
    --eval_batch_size 32
```

Notes:
- For MX data type, `--target_bits` ranges from 4.25 to 8.25 due to scale bits
- `--tune_tasks` indicates the tasks used for autotune accuracy verification, default is the same as `--tasks`.
- `--tune_limit` indicates the selected samples of tasks used for autotune accuracy verification, default is None and uses all samples.
- `--options` indicates the data types used for mix precision.
- `--shared_layer` indicates the layers sharing the same data type for mix precision.


### Llama3 Quantization Recipes

Here we provide several recipes for Llama3 models. The relative accuracy loss of quantized model should be less than 1%.

> Note: You can also enable static quantization for KV cache by adding `--static_kv_dtype fp8` argument to `quantize.py`， or `--static_kv_dtype=fp8` argument to `run_quant.sh` and `run_benchmark.sh`.
>
> You can also enable static quantization for attention by adding `--static_attention_dtype fp8` argument to `quantize.py`， or `--static_attention_dtype=fp8` argument to `run_quant.sh` and `run_benchmark.sh`. When enabled, it automatically sets KV cache dtype to fp8 as well.

#### Llama 3.1 8B MXFP8

RTN (Round-to-Nearest) is enough to keep accuracy.

```bash
# Quantize and export AutoRound format
CUDA_VISIBLE_DEVICES=0 bash run_quant.sh --topology=Llama-3.1-8B --dtype=mxfp8 --input_model=/models/Meta-Llama-3.1-8B-Instruct --output_model=Llama-3.1-8B-MXFP8
```

#### Llama 3.1 8B MXFP4 (Mixed with MXFP8, Target_bits=7.8)

`Target_bits=7.8` is an empirical value.

```bash
CUDA_VISIBLE_DEVICES=0 bash run_quant.sh --topology=Llama-3.1-8B --dtype=mxfp4_mixed --input_model=/models/Meta-Llama-3.1-8B-Instruct --output_model=Llama-3.1-8B-MXFP4-MXFP8
```

To obtain the optimal target bit through `autotune` API:

```bash
CUDA_VISIBLE_DEVICES=0 python quantize.py  \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --quantize \
    --dtype MXFP4 \
    --target_bits 7.2 7.5 7.8 \
    --options "MXFP4" "MXFP8" \
    --shared_layer "k_proj" "v_proj" "q_proj" \
    --shared_layer "gate_proj" "up_proj" \
    --enable_torch_compile  \
    --low_gpu_mem_usage \
    --export_format auto_round  \
    --export_path llama3.1-8B-MXFP4-MXFP8 \
    --tasks mmlu_llama piqa hellaswag gsm8k_llama \
    --eval_batch_size 32
```

#### Llama 3.3 70B MXFP8

RTN (Round-to-Nearest) is enough to keep accuracy.

```bash
CUDA_VISIBLE_DEVICES=0 bash run_quant.sh --topology=Llama-3.3-70B --dtype=mxfp8 --input_model=/models/Llama-3.3-70B-Instruct/ --output_model=Llama-3.3-70B-MXFP8
```

> Note: Within the accuracy threshold, lm_head quantization is acceptable, but this feature is not enabled here to support vLLM inference.

#### Llama 3.3 70B MXFP4 (Mixed with MXFP8, Target_bits=5.8)

`Target_bits=5.8` is an empirical value.

```bash
CUDA_VISIBLE_DEVICES=0 bash run_quant.sh --topology=Llama-3.3-70B --dtype=mxfp4_mixed --input_model=/models/Llama-3.3-70B-Instruct/ --output_model=Llama-3.3-70B-MXFP4-MXFP8
```

#### Llama 3.1 70B MXFP8

RTN (Round-to-Nearest) is enough to keep accuracy.

```bash
CUDA_VISIBLE_DEVICES=0 bash run_quant.sh --topology=Llama-3.1-70B --dtype=mxfp8 --input_model=/models/Llama-3.1-70B-Instruct/ --output_model=Llama-3.1-70B-MXFP8
```

> Note: Within the accuracy threshold, lm_head quantization is acceptable, but this feature is not enabled here to support vLLM inference.

#### Llama 3.1 70B NVFP4

AutoRound tuning helps improve the accuracy.

```bash
CUDA_VISIBLE_DEVICES=0,1 bash run_quant.sh --topology=Llama-3.1-70B --dtype=nvfp4 --input_model=/models/Llama-3.1-70B-Instruct/ --output_model=Llama-3.1-70B-NVFP4
```

> Note: Within the accuracy threshold, lm_head quantization is acceptable, but this feature is not enabled here to support vLLM inference.

#### Llama 3.1 70B uNVFP4

RTN (Round-to-Nearest) is enough to keep accuracy.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_quant.sh --topology=Llama-3.1-70B --dtype=unvfp4 --input_model=/models/Llama-3.1-70B-Instruct/ --output_model=Llama-3.1-70B-uNVFP4
```
Note: If you got OOM issue, either increasing `CUDA_VISIBLE_DEVICES` or reducing `eval_batch_size` is suggested.

## Inference

### MXFP4 & MXFP8

- Both pure MXFP4/MXFP8 and mix-precision model generated by target bits are supported.

#### Prerequisite

```bash
# Install the forked vLLM
git clone -b fused-moe-ar --single-branch --quiet https://github.com/yiliu30/vllm-fork.git && cd vllm-fork
VLLM_USE_PRECOMPILED=1 pip install -e .
```

#### MXFP Benchmark Script

For convenience, we provide a benchmark script that automatically handles GPU detection and tensor parallelism configuration:

**All 5 MXFP benchmark cases:**

1. **Llama 3.1 8B MXFP8** (1 GPU):
```bash
CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh --model_path=Llama-3.1-8B-MXFP8 --gpu_memory_utilization=0.8
```

2. **Llama 3.1 8B MXFP4 Mixed** (1 GPU):
```bash
CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh --model_path=Llama-3.1-8B-MXFP4-MXFP8  --gpu_memory_utilization=0.6
```

3. **Llama 3.3 70B MXFP8** (2 GPU):
```bash
CUDA_VISIBLE_DEVICES=0,1 bash run_benchmark.sh --model_path=Llama-3.3-70B-MXFP8  --gpu_memory_utilization=0.8
```

4. **Llama 3.3 70B MXFP4 Mixed** (2 GPU):
```bash
CUDA_VISIBLE_DEVICES=0,1 bash run_benchmark.sh --model_path=Llama-3.3-70B-MXFP4-MXFP8  --gpu_memory_utilization=0.6
```

5. **Llama 3.1 70B MXFP8** (2 GPU):
```bash
CUDA_VISIBLE_DEVICES=0,1 bash run_benchmark.sh --model_path=Llama-3.1-70B-MXFP8   --gpu_memory_utilization=0.8
```

The script automatically:
- Detects available GPUs from `CUDA_VISIBLE_DEVICES` and sets `tensor_parallel_size` accordingly
- Runs default tasks: `piqa,hellaswag,mmlu_llama,gsm8k_llama` with batch size 64
- Supports custom task selection and batch size adjustment
- Handles special tasks like `mmlu_llama`, `gsm8k_llama` (with chat template) and `longbench` (with extended context length) automatically
- For longbench dataset evaluation, use the `--tasks=longbench` parameter


### NVFP4
NVFP4 is supported by vLLM already, please set `llm_compressor` format for exporting during quantization.

```bash
CUDA_VISIBLE_DEVICES=0 lm_eval --model vllm \
    --model_args pretrained={nvfp4_model_path},tensor_parallel_size=1,data_parallel_size=1 \
    --tasks lambada_openai \
    --batch_size 4
```

### uNVFP4
uNVFP4 is saved in fake format and reloading is not available currently. To verify accuracy after quantization, setting `--accuracy --tasks lambada_openai` in command.

```bash
CUDA_VISIBLE_DEVICES=0 python quantize.py  \
    --model_name_or_path facebook/opt-125m  \
    --quantize \
    --dtype uNVFP4 \
    --enable_torch_compile \
    --low_gpu_mem_usage \
    --export_format fake \
    --export_path OPT-125M-uNVFP4 \
    --accuracy \
    --tasks lambada_openai \
    --eval_batch_size 8  \
    --device_map 0
```
