#!/bin/bash
set -x
device=0

# --quant_vision    ## for vision quantization

CUDA_VISIBLE_DEVICES=$device \
python3 main.py \
--model_name=Qwen/Qwen-VL \
--bits 4 \
--group_size 128 \
--iters 200 \
--seqlen 512 \
--disable_quanted_input \
--image_folder /path/to/coco/images/train2017/ \
--question_file /path/to/Qwen-VL_mix665k.json \
--output_dir "./tmp_autoround"

