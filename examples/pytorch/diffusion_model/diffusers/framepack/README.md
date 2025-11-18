# Step-by-Step

This example quantizes and validates the accuracy of Flux.

# Prerequisite

## 1. Environment

```shell
pip install -r requirements.txt
pip install neural-compressor-pt==3.6
pip install auto-round==0.8.0
git clone https://github.com/Vchitect/VBench.git
cd VBench
pip install -r requirements.txt
pip install vbench
cd ..
git clone https://github.com/lllyasviel/FramePack.git
cd FramePack
pip install -r requirements.txt
cd ..
```

## 2. Prepare Dataset

```shell
cd VBench
sh vbench2_beta_i2v/download_data.sh
```

# Run

## BF16

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash run_benchmark.sh \
    --dataset_location=/path/to/VBench \
    --output_video_path=bf16_video \
    --dimension_list=subject_consistency i2v_background \
```

## MXFP8 or FP8 

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash run_benchmark.sh \
    --scheme=MXFP8 \ # or FP8
    --dataset_location=/path/to/VBench \
    --output_video_path=mxfp8_video \
    --dimension_list=subject_consistency i2v_background \
```

- CUDA_VISIBLE_DEVICES: split the evaluation file into the number of GPUs' subset to speed up the evaluation 
