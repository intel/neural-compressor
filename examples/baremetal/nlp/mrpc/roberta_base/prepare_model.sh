#!/bin/bash
# set -x

# Fine-tune the roberta_base model on the task mrpc
python run_glue.py \
  --model_name_or_path roberta-base \
  --task_name mrpc \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir ./tmp/mrpc/roberta_base/

# export the model after training
python roberta_base_export.py \
   --input_dir=./tmp/mrpc/roberta_base/ \
   --output_model=roberta_base_mrpc.onnx