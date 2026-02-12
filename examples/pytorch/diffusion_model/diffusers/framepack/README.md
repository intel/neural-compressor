# Step-by-Step

This example quantizes and validates the accuracy of Flux.

# Prerequisite

```shell
docker run -d --gpus all -v ... --shm-size=10g --name framepack -it nvcr.io/nvidia/pytorch:25.08-py3 /bin/bash
docker exec -it framepack bash

# install zip according to your system
apt update && apt install zip

git clone https://github.com/intel/neural-compressor.git
cd neural-compressor/examples/pytorch/diffusion_model/diffusers/framepack
# Use `INC_PT_ONLY=1 pip install git+https://github.com/intel/neural-compressor.git@master` for the latest updates before neural-compressor v3.7 release
pip install neural-compressor-pt==3.7
# Use `pip install git+https://github.com/intel/auto-round.git@main` for the latest updates before auto-round v0.9.3 release
pip install auto-round==0.9.3

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
