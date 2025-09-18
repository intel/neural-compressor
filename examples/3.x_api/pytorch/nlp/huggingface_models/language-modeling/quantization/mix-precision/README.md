# Run
 
In this examples, you can verify the accuracy on HPU/CUDA device with emulation of MXFP4, MXFP8, NVFP4 and NVFP4+.

## Quantization

### Demo 

``` python
python quantize.py  --model_name_or_path facebook/opt-125m --quantize --dtype mx_fp4 --batch_size 8 --accuracy
```

> Note: `--dtype` supports `mx_fp4`(MXFP4), `mx_fp8`(MXFP8), `nv_fp4`(NVFP4), `fp4_v2`(NVFP4+)

## Mix-precision Quantization (MXFP4 + MXFP8， Target bits: 6)

```bash
python quantize.py  \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --quantize \
    --dtype mx_fp4 \
    --use_recipe \
    --recipe_file recipes/Meta-Llama-3.1-8B-Instruct_6bits.json \
    --accuracy \
    --batch_size 8




deepspeed --include="localhost:0,1,2,3,4,5,6,7" --master_port=29500 quantize.py  \
    --model_name_or_path /git_lfs/data/pytorch/llama3.3/Meta-Llama-3.3-70B-Instruct/ \
    --quantize \
    --dtype mx_fp4 \
    --use_recipe \
    --recipe_file recipes/Meta-Llama-3.3-70B-Instruct_6bits.json \
    --accuracy \
    --batch_size 2 2>&1|tee 70b.log


```

