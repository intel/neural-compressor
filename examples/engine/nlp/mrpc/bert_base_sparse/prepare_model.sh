#!/bin/bash
# set -x

# Fine-tune the bert_base_sparse model on the task MRPC
python run_glue.py \
  --model_name_or_path Intel/bert-base-uncased-sparse-70-unstructured \
  --task_name mrpc \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --warmup_ratio 0.2 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --output_dir ./tmp/mrpc/bert_base_sparse/

# export the model after training
python bert_base_sparse_export.py \
   --input_dir=./tmp/mrpc/bert_base_sparse/ \
   --task_name=MRPC \
   --output_model=bert_base_sparse_mrpc.onnx

cp ./tmp/mrpc/bert_base_sparse/vocab.txt ./data/.