#!/bin/bash
set -x
device=0
model_name=microsoft/Phi-3-vision-128k-instruct
CUDA_VISIBLE_DEVICES=$device \
python3 main.py \
--model_name=$model_name \
--nsamples 512 \
--image_folder /PATH/TO/coco/images/train2017 \
--question_file /PATH/TO/llava_v1_5_mix665k.json \
--output_dir "./tmp_autoround"


