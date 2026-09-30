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

### Prompt Tests

Usage: 
```bash
bash ./run_generate.sh -s [mxfp4|mxfp8] -tp [tensor_parallel_size] -m [model_path]
```

- MXFP8
```bash
bash ./run_generate.sh -s mxfp8 -tp 4 -m /path/to/qwen_mxfp8
```
- MXFP4
```bash
bash ./run_generate.sh -s mxfp4 -tp 4 -m /path/to/qwen_mxfp4
```
- KV Cache
```bash
bash ./run_generate.sh -s mxfp4 -tp 1 -kv fp8 -m /path/to/qwen_mxfp4
```
### Evaluation


Usage: 
```bash
bash run_benchmark.sh --model_path=<model_path> --scheme=[mxfp4|mxfp8] --tasks=<task_name> --batch_size=<size>
```
- MXFP8
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_benchmark.sh --model_path=/path/to/qwen_mxfp8 --scheme=mxfp8 --tasks=piqa,hellaswag,mmlu,gsm8k --batch_size=256

```
- MXFP4
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_benchmark.sh --model_path=/path/to/qwen_mxfp4 --scheme=mxfp4 --tasks=piqa,hellaswag,mmlu,gsm8k --batch_size=256
```