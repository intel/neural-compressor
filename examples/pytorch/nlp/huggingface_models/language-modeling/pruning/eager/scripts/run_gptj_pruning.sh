#!/bin/bash

set -x

# Set environment
CUBLAS_WORKSPACE_CONFIG=':4096:8'


CUDA_VISIBLE_DEVICES=4 python \
    examples/pytorch/nlp/huggingface_models/language-modeling/pruning/eager/run_clm_no_trainer.py \
    --model_name_or_path EleutherAI/gpt-j-6b \
    --calibration_dataset_name NeelNanda/pile-10k \
    --evaluation_dataset_name lambada \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 16 \
    --max_pruning_steps 3002 \
    --weight_decay  0 \
    --block_size 512 \
    --max_length 512 \
    --do_prune \
    --auto_slim \
    --output_dir ./sparse_model \
    --target_sparsity 0.1 \
    --pruning_pattern channelx1 \
    --pruning_frequency 500
    
