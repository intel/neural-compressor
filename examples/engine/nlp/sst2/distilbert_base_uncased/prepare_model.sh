#!/bin/bash
# set -x

# Fine-tune the distilbert_base_uncased model on the task SST-2
python run_glue.py \
  --model_name_or_path distilbert-base-uncased-finetuned-sst-2-english \
  --task_name sst2 \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --output_dir ./tmp/sst2/distilbert_base_uncased/

# export the model after training
python distilbert_base_export.py \
   --input_dir=./tmp/sst2/distilbert_base_uncased/ \
   --task_name=SST-2 \
   --output_model=distilbert_base_uncased_sst2.onnx