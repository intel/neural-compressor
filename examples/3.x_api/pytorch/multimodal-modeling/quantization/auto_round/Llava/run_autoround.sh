#!/bin/bash
set -x
device=0

CUDA_VISIBLE_DEVICES=$device \
python3 main.py \
--model_name=liuhaotian/llava-v1.5-7b \
--bits 4 \
--group_size 128 \
--iters 200 \
--seqlen 512 \
--quantize \
--image_folder /path/to/coco/images/train2017/ \
--question_file /path/to/LLaVA-Instruct-150K/llava_v1_5_mix665k.json \
--eval-question-file /path/to/textvqa/llava_textvqa_val_v051_ocr.jsonl \
--eval-image-folder /path/to/textvqa/train_images \
--eval-annotation-file /path/to/textvqa/TextVQA_0.5.1_val.json \
--eval-result-file "./tmp_autoround" \
--output_dir "./tmp_autoround"


