#!/bin/bash
# set -x

# Fine-tune the distilroberta-base model on the task cola
python run_glue.py \
  --model_name_or_path distilroberta-base \
  --task_name wnli \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir ./tmp/wnli/distilroberta_base/

# export the model after training
python distilroberta_base_export.py \
   --input_dir=./tmp/wnli/distilroberta_base/ \
   --task_name=WNLI \
   --output_model=distilroberta_base_wnli.onnx
