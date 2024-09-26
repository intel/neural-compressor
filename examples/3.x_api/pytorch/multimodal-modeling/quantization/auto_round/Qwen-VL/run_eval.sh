#!/bin/bash
set -x
device=0

model_path='./tmp_autoround'
model=Qwen-VL

CUDA_VISIBLE_DEVICES=$device python3 mm_evaluation/main.py \
--model_name ${model_path}/${model} \
--trust_remote_code \
--eval_bs 4




