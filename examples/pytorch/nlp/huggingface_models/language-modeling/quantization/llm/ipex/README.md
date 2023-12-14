Step-by-Step
============
This document describes the step-by-step instructions to run llama2 SmoothQuant with Intel® Neural Compressor and Intel® Extension for PyTorch.

# Prerequisite
```
# Installation dependecies
pip install -r requirements.txt
```

# Run Quantization

## Llama-2-7b
```bash
python run_llama2_sq.py \
    --model-id meta-llama/Llama-2-7b-hf \
    --batch-size 56 \
    --sq-recipes "llama2-7b"
```
## Llama-2-13b
```bash
python run_llama2_sq.py \
    --model-id meta-llama/Llama-2-13b-hf \
    --batch-size 56 \
    --sq-recipes "llama2-13b"
```
> Notes:  
> INT8 model will be saved into "./saved_results". 
> parameter "--sq-recipes" decides the recipes use to do quantize, details can be found in scripts.