#!/bin/bash
set -x

# Set environment
export CUBLAS_WORKSPACE_CONFIG=':4096:8'

# Available Models

# Common Large Language Models(LLMs), e.g. OPT, GPT, LLaMA, BLOOM, Dolly, MPT, Falcon, Stable-LM, LaMini-LM, etc.

#cd neural-compressor
python examples/pytorch/nlp/huggingface_models/language-modeling/pruning/eager/run_clm_sparsegpt.py \
    --model_name_or_path /PATH/TO/LLM/ \
    --do_prune \
    --device=0 \
    --output_dir=/PATH/TO/SAVE/ \
    --eval_dtype 'bf16' \
    --per_device_eval_batch_size 16 \
    --target_sparsity 0.5 \
    --pruning_pattern 1x1

