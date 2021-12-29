#!/bin/bash
# set -x

# Fine-tune the bert_base_nli_mean_tokens model on the task stsb  
python run_glue.py \
  --model_name_or_path sentence-transformers/bert-base-nli-mean-tokens \
  --task_name stsb \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 5 \
  --output_dir ./tmp/stsb/bert_base_nli_mean_tokens/

# export the model after training
python bert_base_nli_export.py \
   --input_dir=./tmp/stsb/bert_base_nli_mean_tokens/ \
   --task_name=STS-B \
   --output_model=bert_base_nli_mean_tokens_stsb.onnx