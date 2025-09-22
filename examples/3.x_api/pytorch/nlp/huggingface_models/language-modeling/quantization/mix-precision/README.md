# Run
 
In this examples, you can verify the accuracy on HPU/CUDA device with emulation of MXFP4, MXFP8, NVFP4 and NVFP4+.

## Requirement

```bash
# neural-compressor-pt
INC_PT_ONLY=1 pip install git+https://github.com/intel/neural-compressor.git@xinhe/mx_recipe
# auto-round
pip install git+https://github.com/intel/auto-round.git@xinhe/llama_tmp
```

## Quantization

### Demo 

``` python
python quantize.py  --model_name_or_path facebook/opt-125m --quantize --dtype mx_fp4 --batch_size 8 --accuracy
```

> Note: `--dtype` supports `mx_fp4`(MXFP4), `mx_fp8`(MXFP8), `nv_fp4`(NVFP4), `fp4_v2`(NVFP4+)

## Mix-precision Quantization (MXFP4 + MXFP8， Target bits: 6)

```bash
# Llama 3.1 8B
python quantize.py  \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --quantize \
    --dtype mx_fp4 \
    --use_recipe \
    --recipe_file recipes/Meta-Llama-3.1-8B-Instruct_7bits.json \
    --accuracy \
    --batch_size 32


# Llama 3.3 70B
deepspeed --include="localhost:4,5,6,7" --master_port=29500 quantize.py  \
    --model_name_or_path meta-llama/Llama-3.3-70B-Instruct/ \
    --quantize \
    --dtype mx_fp4 \
    --use_recipe \
    --recipe_file recipes/Meta-Llama-3.3-70B-Instruct_6bits.json \
    --accuracy \
    --batch_size 32
```

