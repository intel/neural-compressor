# Run
 
In this examples, you can verify the accuracy on HPU/CUDA device with emulation of MXFP4, MXFP8, NVFP4 and uNVFP4.

## Requirement

```bash
# neural-compressor-pt
INC_PT_ONLY=1 pip install git+https://github.com/intel/neural-compressor.git@v3.6
# auto-round
pip install git+https://github.com/intel/auto-round.git@v0.8.0
```
or
```bash
# neural-compressor-pt
INC_PT_ONLY=1 pip install git+https://github.com/intel/neural-compressor.git@xinhe/mx_recipe
# auto-round
pip install git+https://github.com/intel/auto-round.git@xinhe/llama_tmp
```

## Quantization

### Demo (`MXFP4`, `MXFP8`, `NVFP4`, `uNVFP4`)

```bash
python quantize.py  --model_name_or_path facebook/opt-125m --quantize --dtype MXFP4 --batch_size 8 --accuracy
```


### Mix-precision Quantization (`MXFP4 + MXFP8`)

```bash
# Llama 3.1 8B
python quantize.py  \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --quantize \
    --dtype MXFP4 \
    --use_recipe \
    --recipe_file recipes/Meta-Llama-3.1-8B-Instruct_7bits.json \
    --accuracy \
    --batch_size 32

# Llama 3.3 70B
deepspeed --include="localhost:4,5,6,7" --master_port=29500 python quantize.py  \
    --model_name_or_path meta-llama/Llama-3.3-70B-Instruct/ \
    --quantize \
    --dtype MXFP4 \
    --use_recipe \
    --recipe_file recipes/Meta-Llama-3.3-70B-Instruct_5bits.json \
    --accuracy \
    --batch_size 32
```

> Note: 
> 1. Quantization applies `--dtype` for all blocks in the model by removing `--use_recipe`.
> 2. Setting `--quant_lm_head` applies `--dtype` for the lm_head layer.
> 3. Setting `--iters 0` skips AutoRound tuning and uses RTN method.
> 4. The `deepspeed` usage provides quick accuracy verification. As for saving， please use [`--device_map`](#mxfp4--mxfp8).

## Inference usage

### NVFP4
NVFP4 is supported by vLLM already, the saved model in this example follows the `llm_compressor` format, please refer to the usage in the public vLLM document.

```bash
# Command to save model:
python quantize.py  --model_name_or_path facebook/opt-125m --quantize --dtype NVFP4 --batch_size 8 --save --save_path opt-125m-nvfp4 --save_format llm_compressor
```

### MXFP4 / MXFP8

MXFP4 is enabled in a forked vLLM repo, usages as below:
```bash
# Install the forked vLLM for MXFP4

# Command to save model:
python quantize.py  --model_name_or_path facebook/opt-125m --quantize --dtype MXFP4 --batch_size 8 --save --save_path opt-125m-mxfp4 --save_format llm_compressor

# Command to evaluate with vLLM:

```

### MXFP4 + MXFP8
Model with mixed precision is not supported in vLLM, but supported in transformers in auto-round format. `--device_map 0,1,2` provides pipeline parallel so that we can save the model without tensor sharding.

```bash
# Command to save model:
python quantize.py  \
    --model_name_or_path /models/Llama-3.3-70B-Instruct/ \
    --quantize \
    --device_map 0,1,2 \
    --dtype MXFP4 \
    --use_recipe \
    --recipe_file recipes/Meta-Llama-3.3-70B-Instruct_5bits.json \
    --save \
    --save_format auto_round \
    --save_path llama3.3-70b-mxfp4-mxfp8
# Command to evaluate with transformer:

```
