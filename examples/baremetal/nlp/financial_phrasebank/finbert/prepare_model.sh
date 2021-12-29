#!/bin/bash
# set -x

# Fine-tune the finbert model on the dataset financial_phrasebank
python run_glue.py \
  --model_name_or_path ProsusAI/finbert \
  --dataset_name financial_phrasebank \
  --dataset_config_name sentences_50agree \
  --do_train \
  --do_eval \
  --max_seq_length 64 \
  --warmup_ratio 0.2 \
  --per_device_train_batch_size 64 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --output_dir ./tmp/finbert/

# export the model after training
python finbert_export.py \
   --input_dir=./tmp/finbert/ \
   --output_model=finbert_financial_phrasebank.onnx