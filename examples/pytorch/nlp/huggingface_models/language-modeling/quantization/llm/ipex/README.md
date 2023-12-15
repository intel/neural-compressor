Step-by-Step
============
This document describes the step-by-step instructions to run llama2 SmoothQuant with Intel® Neural Compressor and Intel® Extension for PyTorch.

# Prerequisite
```
# Installation dependencies
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
    --sq-recipes "llama2-13b" \
    --padding
```
> Notes:  
> - INT8 model will be saved into "./saved_results" including "./saved_results/best_configure.json" and "./saved_results/best_model.pt", which can be loaded and evaluated by IPEX.  
> - Parameter "--sq-recipes" decides the recipes used to do quantize, details can be found in scripts.