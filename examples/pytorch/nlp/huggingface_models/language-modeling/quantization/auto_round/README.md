
## Support Matrix

| Model Family            | MXFP4 | MXFP8 |
| ----------------------- | ----- | ----- |
| Qwen/Qwen3-235B-A22B    | ✅     | ✅     |
| deepseek-ai/DeepSeek-R1 | ✅     | ✅     |

### Quantize Model
- Export model path
```bash
export QWEN_MODEL=Qwen/Qwen3-235B-A22B
export QWEN_MODEL=/storage/yiliu7/Qwen/Qwen3-30B-A3B-Base/
export DS_MODEL=deepseek-ai/DeepSeek-R1
```

- MXFP8
```bash
python quantize.py --model $QWEN_MODEL -t qwen_mxfp8 --use_autoround_format --output_dir ./qmodels
python quantize.py --model $DS_MODEL -t ds_mxfp8 --use_autoround_format ----output_dir ./qmodels
```
/storage/yiliu7/meta-llama/Meta-Llama-3-8B-Instruct
- MXFP4
```bash
python quantize.py --model $QWEN_MODEL -t qwen_mxfp4 --use_autoround_format --output_dir ./qmodels
python quantize.py --model $DS_MODEL -t qwen_mxfp4 --use_autoround_format --output_dir ./qmodels
```


### Prompt Tests

Usage: 
```bash
bash ./run_generate.sh -s [mxfp4|mxfp8] -tp [tensor_parallel_size] -m [model_path]
```

- MXFP8
```bash
bash ./run_generate.sh -s mxfp8 -tp 4 -m /path/to/qwen_mxfp8
bash ./run_generate.sh -s mxfp8 -tp 8 -m /path/to/ds_mxfp8
```
- MXFP4
```bash
bash ./run_generate.sh -s mxfp4 -tp 4 -m /path/to/qwen_mxfp 
bash ./run_generate.sh -s mxfp4 -tp 8  -m /path/to/ds_mxfp4
```
### Evaluation Tests

WIP



