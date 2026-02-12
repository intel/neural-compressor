# Step-by-Step

This example quantizes and validates the accuracy of Flux.

# Prerequisite

## 1. Environment

```shell
pip install -r requirements.txt
# Use `INC_PT_ONLY=1 pip install git+https://github.com/intel/neural-compressor.git@master` for the latest updates before neural-compressor v3.6 release
pip install neural-compressor-pt==3.7
# Use `pip install git+https://github.com/intel/auto-round.git@main` for the latest updates before auto-round v0.8.0 release
pip install auto-round==0.9.3
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

## Quantization

```bash
bash run_quant.sh --topology=flux_mxfp8 --input_model=FLUX.1-dev --output_model=mxfp8_model
```
- topology: support flux_fp8 and flux_mxfp8


## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_benchmark.sh --topology=flux_mxfp8 --input_model=FLUX.1-dev --quantized_model=mxfp8_model
```

- CUDA_VISIBLE_DEVICES: split the evaluation file into the number of GPUs' subset to speed up the evaluation 
