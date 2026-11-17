This example provides an end-to-end workflow to quantize Qwen models to MXFP4/MXFP8 and evaluate them using a custom vLLM fork.

## Requirement
```bash
uv pip install neural-compressor-pt
uv pip install auto-round
bash setup.sh
```

### Quantize Model
- Export model path
```bash
export MODEL=Qwen/Qwen3-235B-A22B
```
> [!TIP]
> For quicker experimentation (shorter quantization and evaluation time, lower memory),
> you can start with the smaller `export MODEL=Qwen/Qwen3-30B-A3B` model before moving to larger variants.
> Currently, KV cache quantization only supports **FP8**.


- MXFP8
```bash
bash run_quant.sh --dtype=mxfp8 --input_model=$MODEL --output_model=./qmodels
```

- MXFP4
```bash
bash run_quant.sh --dtype=mxfp4 --input_model=$MODEL --output_model=./qmodels
```
- KV Cache
```bash
export MODEL=Qwen/Qwen3-30B-A3B
bash run_quant.sh --dtype=mxfp4 --input_model=$MODEL --output_model=./qmodels --static_kv_dtype=fp8
```

  Attention
```bash
export MODEL=Qwen/Qwen3-30B-A3B
bash run_quant.sh --dtype=mxfp4 --input_model=$MODEL --output_model=./qmodels --static_attention_dtype=fp8
```

## Evaluation

Usage:
```bash
CUDA_VISIBLE_DEVICES=<gpu_ids> bash run_benchmark.sh --model_path=<model_path> [--tasks=<task1,task2,...>]
```

The default task list is `hellaswag,piqa,mmlu,gsm8k`. `aime26`, `gpqa_diamond` and `math_500` run with `benchmark/evalscope/run_evalscope.sh`; all other tasks run with `benchmark/lm_eval/run_lm_eval.sh`. Tensor parallel size is inferred from `CUDA_VISIBLE_DEVICES`.

- MXFP8
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_benchmark.sh --model_path=/path/to/qwen_mxfp8 --tasks=piqa,hellaswag,mmlu,gsm8k
```
- MXFP4
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_benchmark.sh --model_path=/path/to/qwen_mxfp4 --tasks=piqa,hellaswag,mmlu,gsm8k
```
- KV Cache / Attention
```bash
CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh --model_path=/path/to/qwen_mxfp4 --static_kv_dtype=fp8
CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh --model_path=/path/to/qwen_mxfp4 --static_attention_dtype=fp8
```
- evalscope tasks
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_benchmark.sh --model_path=/path/to/qwen_mxfp4 --tasks=aime26,gpqa_diamond,math_500
```
