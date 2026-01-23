This example provides an end-to-end workflow to quantize Qwen models to MXFP4/MXFP8 and evaluate them using a custom vLLM fork.

## Requirement
```bash
pip install neural-compressor-pt
# auto-round
pip install auto-round
# vLLM
git clone -b fused-moe-ar --single-branch --quiet https://github.com/yiliu30/vllm-fork.git && cd vllm-fork
VLLM_USE_PRECOMPILED=1 pip install --editable . -vvv
# other requirements
pip install -r requirements.txt
pip uninstall flash_attn
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
bash run_quant.sh --model $MODEL -t mxfp8 --output_dir ./qmodels
```

- MXFP4
```bash
bash run_quant.sh --model $MODEL -t mxfp4 --output_dir ./qmodels
```
- KV Cache
```bash
export MODEL=Qwen/Qwen3-30B-A3B
bash run_quant.sh --model $MODEL -t mxfp4 --output_dir ./qmodels -kv "fp8"
```

  Attention
```bash
export MODEL=Qwen/Qwen3-30B-A3B
bash run_quant.sh --model $MODEL -t mxfp4 --output_dir ./qmodels -attn "fp8"
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
bash run_evaluation.sh -m [model_path] -s [mxfp4|mxfp8] -t [task_name] -tp [tensor_parallel_size] -b [batch_size]
```
```bash
bash run_evaluation.sh -s mxfp8 -t piqa,hellaswag,mmlu -tp 4 -b 512 -m /path/to/qwen_mxfp8
bash run_evaluation.sh -s mxfp8 -t gsm8k -tp 4 -b 256 -m /path/to/qwen_mxfp8

```
- MXFP4
```bash
bash run_evaluation.sh -s mxfp4 -t piqa,hellaswag,mmlu -tp 4 -b 512 -m /path/to/qwen_mxfp4
bash run_evaluation.sh -s mxfp4 -t gsm8k -tp 4 -b 256 -m /path/to/qwen_mxfp4
```