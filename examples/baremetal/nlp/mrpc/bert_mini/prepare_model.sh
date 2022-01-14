#!/bin/bash
# set -x

# Fine-tune the bert_mini model on the task MRPC
python run_glue.py \
  --model_name_or_path prajjwal1/bert-mini \
  --task_name mrpc \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --warmup_ratio 0.2 \
  --per_device_train_batch_size 128 \
  --learning_rate 8e-5 \
  --num_train_epochs 15 \
  --output_dir ./tmp/mrpc/bert_mini/ \
  --overwrite_output_dir

# export the model after training
python bert_mini_export.py \
   --input_dir=./tmp/mrpc/bert_mini/ \
   --output_model=bert_mini_mrpc.onnx