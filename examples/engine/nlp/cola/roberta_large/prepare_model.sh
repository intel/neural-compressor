#!/bin/bash
# set -x

# Fine-tune the roberta_large model on the task cola
python run_glue.py \
  --model_name_or_path roberta-large \
  --task_name cola \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --warmup_ratio 0.06 \
  --output_dir ./tmp/cola/roberta_large/
  --overwrite_output_dir

# export the model after training
python roberta_large_export.py \
   --input_dir=./tmp/cola/roberta_large/ \
   --task_name=COLA \
   --output_model=roberta_large_cola.onnx
