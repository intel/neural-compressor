#!/bin/bash
# set -x

# Fine-tune the distilbert_base_uncased  model on the task MRPC
python run_glue.py \
  --model_name_or_path distilbert-base-uncased \
  --task_name mrpc \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir ./tmp/mrpc/distilbert_base_uncased/

# export the model after training
python distilbert_base_export.py \
   --input_dir=./tmp/mrpc/distilbert_base_uncased/ \
   --task_name=MRPC \
   --output_model=distilbert_base_uncased_mrpc.onnx

cp ./tmp/mrpc/distilbert_base_uncased/vocab.txt ./data/.