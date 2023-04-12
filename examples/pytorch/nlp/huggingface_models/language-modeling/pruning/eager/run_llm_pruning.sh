#!/bin/bash
set -x
    python \
    examples/pytorch/nlp/huggingface_models/language-modeling/pruning/eager/run_clm_no_trainer.py \
    --model_name_or_path /path/to/your/model \
    --dataset_name lambada \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 16 \
    --max_train_steps 3002 \
    --weight_decay  0 \
    --block_size 512 \
    --do_prune \
    --auto_slim \
    --output_dir sparse_clm_models/ \
    --target_sparsity 0.2 \
    --pruning_pattern channelx1 \
    --pruning_frequency 500 \
