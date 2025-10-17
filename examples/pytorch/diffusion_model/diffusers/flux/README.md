# Step-by-Step

This example quantizes and validates the accuracy of Flux.

# Prerequisite

## 1. Environment

```shell
pip install -r requirements.txt
# Use `INC_PT_ONLY=1 pip install git+https://github.com/intel/neural-compressor.git@v3.6rc` for the latest updates before neural-compressor v3.6 release
pip install neural-compressor-pt==3.6
# Use `pip install git+https://github.com/intel/auto-round.git@v0.8.0rc2` for the latest updates before auto-round v0.8.0 release
pip install auto-round==0.8.0
```

## 2. Prepare Model

```shell
hf download black-forest-labs/FLUX.1-dev --local-dir FLUX.1-dev
```

## 3. Prepare Dataset
```shell
wget https://github.com/mlcommons/inference/raw/refs/heads/master/text_to_image/coco2014/captions/captions_source.tsv
```

# Run

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_quant.sh --topology=flux_fp8 --input_model=FLUX.1-dev
```
- topology: support flux_fp8 and flux_mxfp8
- CUDA_VISIBLE_DEVICES: split the evaluation file into the number of GPUs' subset to speed up the evaluation 
