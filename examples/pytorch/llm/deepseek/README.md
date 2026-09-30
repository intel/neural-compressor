This example provides an end-to-end workflow to quantize DeepSeek models to MXFP4/MXFP8/NVFP4 and evaluate them using a custom vLLM fork.

## Requirement
```bash
pip install neural-compressor-pt
pip install auto-round
bash setup.sh
```

### Quantize Model
- Export model path
```bash
export MODEL=unsloth/DeepSeek-R1-BF16
```

- MXFP8
```bash
bash run_quant.sh --dtype=mxfp8 --input_model=$MODEL --output_model=./qmodels
```

- MXFP4
```bash
bash run_quant.sh --dtype=mxfp4 --input_model=$MODEL --output_model=./qmodels
```

- NVFP4
```bash
bash run_quant.sh --dtype=nvfp4 --input_model=$MODEL --output_model=./qmodels
```

To enable `fp8 kv cache`, please add `--static_kv_dtype=fp8`:
```bash
# w/ fp8 kv
bash run_quant.sh --dtype=mxfp4 --input_model=$MODEL --output_model=./qmodels --static_kv_dtype=fp8
```

  Attention
```bash
export MODEL=unsloth/DeepSeek-R1-BF16
bash run_quant.sh --dtype=mxfp4 --input_model=$MODEL --output_model=./qmodels --static_attention_dtype=fp8
```

## Evaluation

### Prompt Tests

Usage: 
```bash
bash ./run_generate.sh -s [mxfp4|mxfp8|nvfp4] -tp [tensor_parallel_size] -m [model_path]
```

- MXFP8
```bash
bash ./run_generate.sh -s mxfp8 -tp 8 -m /path/to/ds_mxfp8
```
- MXFP4
```bash
bash ./run_generate.sh -s mxfp4 -tp 8 -m /path/to/ds_mxfp4
```
- NVFP4
```bash
bash ./run_generate.sh -s nvfp4 -tp 8 -m /path/to/ds_nvfp4
```
### Evaluation


Usage: 
```bash
bash run_benchmark.sh --model_path=<model_path> --scheme=[mxfp4|mxfp8|nvfp4] --tasks=<task_name> --batch_size=<size>
```
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh --model_path=/path/to/ds_mxfp8 --scheme=mxfp8 --tasks=piqa,hellaswag,mmlu,gsm8k --batch_size=256

```
- MXFP4
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh --model_path=/path/to/ds_mxfp4 --scheme=mxfp4 --tasks=piqa,hellaswag,mmlu,gsm8k --batch_size=256
```
- NVFP4
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh --model_path=/path/to/ds_nvfp4 --scheme=nvfp4 --tasks=piqa,hellaswag,mmlu,gsm8k --batch_size=256
```