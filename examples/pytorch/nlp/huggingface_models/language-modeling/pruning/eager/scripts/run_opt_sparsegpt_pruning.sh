#!/bin/bash

set -x
CUDA_VISIBLE_DEVICES=4 python \
    examples/pytorch/nlp/huggingface_models/language-modeling/pruning/eager/run_clm_no_trainer_sparsegpt.py \
    --model_name_or_path facebook/opt-1.3b \
    --calibration_dataset_name wikitext2 \
    --evaluation_dataset_name wikitext2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 16 \
    --max_pruning_steps 3002 \
    --weight_decay  0 \
    --block_size 512 \
    --max_length 512 \
    --do_prune \
    --auto_slim \
    --output_dir ./sparse_model \
    --target_sparsity 0.5 \
    --pruning_pattern 2:4 \
    --pruning_frequency 500
    
    