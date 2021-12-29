#!/bin/bash
# set -x

# git clone the fine-tuned minilm model on task sst2
git lfs clone https://huggingface.co/philschmid/MiniLM-L6-H384-uncased-sst2

# export the model
python minilm_export.py \
   --input_dir=./MiniLM-L6-H384-uncased-sst2 \
   --task_name=SST-2 \
   --output_model=minilm_l6_h384_uncased_sst2.onnx