# Step-by-Step

This example quantizes and validates the accuracy of Flux.

# Prerequisite

```shell
# install zip according to your system
sudo apt update && sudo apt install zip

bash setup.sh

# several models will be downloaded automatically into HF_HOME
export HF_HOME=/path/to/save/model
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
