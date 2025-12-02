This example provides an end-to-end workflow to quantize DeepSeek models to MXFP4/MXFP8 and evaluate them using a custom vLLM fork.

## Requirement
```bash
pip install neural-compressor-pt==3.7
# auto-round
pip install auto-round==0.9.1
# vLLM
git clone -b fused-moe-ar --single-branch --quiet https://github.com/yiliu30/vllm-fork.git && cd vllm-fork
VLLM_USE_PRECOMPILED=1 pip install --editable . -vvv
# other requirements
pip install -r requirements.txt
```

### Quantize Model
- Export model path
```bash
export MODEL=deepseek-ai/DeepSeek-R1
```

- MXFP8
```bash
bash run_quant.sh --model $MODEL -t mxfp8 --output_dir ./qmodels
```

- MXFP4
```bash
bash run_quant.sh --model $MODEL -t mxfp8 --output_dir ./qmodels
```

## Evaluation

### Prompt Tests

Usage: 
```bash
bash ./run_generate.sh -s [mxfp4|mxfp8] -tp [tensor_parallel_size] -m [model_path]
```

- MXFP8
```bash
bash ./run_generate.sh -s mxfp8 -tp 8 -m /path/to/ds_mxfp8
```
- MXFP4
```bash
bash ./run_generate.sh -s mxfp4 -tp 8 -m /path/to/ds_mxfp 
```
### Evaluation


Usage: 
```bash
bash run_evaluation.sh -m [model_path] -s [mxfp4|mxfp8] -t [task_name] -tp [tensor_parallel_size] -b [batch_size]
```
```bash
bash run_evaluation.sh -s mxfp8 -t piqa,hellaswag,mmlu -tp 8 -b 512 -m /path/to/ds_mxfp8
bash run_evaluation.sh -s mxfp8 -t gsm8k -tp 8 -b 256 -m /path/to/ds_mxfp8

```
- MXFP4
```bash
bash run_evaluation.sh -s mxfp4 -t piqa,hellaswag,mmlu -tp 8 -b 512 -m /path/to/ds_mxfp4
bash run_evaluation.sh -s mxfp4 -t gsm8k -tp 8 -b 256 -m /path/to/ds_mxfp4
```