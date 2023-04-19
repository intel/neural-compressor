#!/bin/bash

#### The following types of model pruning are currently supported ####
#   TYPE    EXAMPLE
#   opt     facebook/opt-1.3b
#   bloom   bigscience/bloom-3b
#   gptj    EleutherAI/gpt-j-6b
#   llama   decapoda-research/llama-7b-hf

set -x
python \
    examples/pytorch/nlp/huggingface_models/language-modeling/pruning/eager/run_clm_no_trainer.py \
    --model_name_or_path /path/to/llm_model/ \
    --calibration_dataset_name /path/to/dataset/the_pile/ \
    --evaluation_dataset_name /path/to/dataset/lambada \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 16 \
    --max_pruning_steps 3002 \
    --weight_decay  0 \
    --block_size 512 \
    --do_prune \
    --auto_slim \
    --output_dir /sparse_model/ \
    --target_sparsity 0.1 \
    --pruning_pattern channelx1 \
    --pruning_frequency 500
