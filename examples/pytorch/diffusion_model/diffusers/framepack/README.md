# Step-by-Step

This example quantizes and validates the accuracy of Flux.

# Prerequisite

## 1. Environment

```shell
# install zip according to your system
sudo apt update && sudo apt install zip

pip install -r requirements.txt
pip install --update neural-compressor-pt
pip install --update auto-round
git clone --depth 1 https://github.com/lllyasviel/FramePack.git
cd FramePack
git fetch origin 97fe5dbe06ac1f337ece08935b1076a35eefeeb9 --depth=1
git reset --hard FETCH_HEAD
cd ..
cp -r FramePack/diffusers_helper/ .

# several models will be downloaded automatically into HF_HOME
export HF_HOME=/path/to/save/model
```

## 2. Prepare Dataset

```shell
git clone --depth 1 https://github.com/Vchitect/VBench.git
cd VBench
git fetch origin 07bc8a4b74d5e0a23de42ed5880b899a1ff705f0 --depth=1
git reset --hard FETCH_HEAD
sh vbench2_beta_i2v/download_data.sh
```

# Run

## BF16

```bash
CUDA_VISIBLE_DEVICES=0,1 \
bash run_benchmark.sh \
    --topology=BF16 \
    --dataset_location=/path/to/VBench \
    --output_video_path=bf16_video \
    --dimension_list="subject_consistency i2v_background" \
    --result_path=bf16_result
```

## MXFP8 or FP8 

```bash
CUDA_VISIBLE_DEVICES=0,1 \
bash run_benchmark.sh \
    --topology=MXFP8 \ # or FP8
    --dataset_location=/path/to/VBench \
    --output_video_path=mxfp8_video \
    --dimension_list="subject_consistency i2v_background" \
    --result_path=mxfp8_result
```

- CUDA_VISIBLE_DEVICES: distribute the data to different visible GPUs to speed up the evaluation
