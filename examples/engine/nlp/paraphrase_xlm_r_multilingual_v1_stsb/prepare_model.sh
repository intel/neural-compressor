#!/bin/bash
# set -x

# Fine-tune the paraphrase-xlm-r-multilingual-v1 model on the task stsb  
python run_glue.py \
  --model_name_or_path sentence-transformers/paraphrase-xlm-r-multilingual-v1  \
  --task_name stsb \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 5 \
  --output_dir ./tmp/stsb/paraphrase_xlm_r_multilingual_v1/

# export the model after training
python xlm_roberta_base_export.py \
   --input_dir=./tmp/stsb/paraphrase_xlm_r_multilingual_v1/ \
   --task_name=STS-B \
   --output_model=paraphrase_xlm_r_multilingual_v1_stsb.onnx