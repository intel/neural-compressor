
### Quantize Model
- Export model path
```bash
export MODEL=Qwen/Qwen3-235B-A22B
```

- MXFP8
```bash
python quantize.py --model $MODEL -t mxfp8 --use_autoround_format --output_dir ./qmodels
```

- MXFP4
```bash
bash run_quant.sh --model "$MODEL" -t mxfp8 --output_dir ./qmodels
```

## Evaluation
```bash
git clone https://github.com/yiliu30/vllm-fork/tree/
cd vllm-fork
git checkout fused-moe-ar
VLLM_USE_PRECOMPILED=1 pip install --editable . -vvv
```

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
bash ./run_generate.sh -s mxfp4 -tp 4 -m /path/to/qwen_mxfp 
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




