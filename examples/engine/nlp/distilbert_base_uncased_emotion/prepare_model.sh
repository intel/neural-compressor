#!/bin/bash
# set -x

# Fine-tune the distilbert_base_uncased model on the emotion dataset 
python run_trainer.py \
  --model_name_or_path bhadresh-savani/distilbert-base-uncased-emotion  \
  --dataset_name emotion \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 64 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir ./tmp/emotion/distilbert_base_uncased/

# export the model after training
python distilbert_base_export.py \
   --input_dir=./tmp/emotion/distilbert_base_uncased/ \
   --output_model=distilbert_base_uncased_emotion.onnx