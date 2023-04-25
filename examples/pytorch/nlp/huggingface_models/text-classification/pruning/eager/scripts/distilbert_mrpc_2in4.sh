#!/bin/bash
set -x

    python examples/pytorch/nlp/huggingface_models/text-classification/pruning/eager/run_glue_no_trainer.py \
        --model_name_or_path "path/to/distilbert-base-uncased-mrpc/dense_finetuned_model" \
        --task_name "mrpc" \
        --max_length 256 \
        --per_device_train_batch_size 16 \
        --learning_rate 1e-4\
        --num_train_epochs 120 \
        --weight_decay 0 \
        --cooldown_epochs 40 \
        --sparsity_warm_epochs 1 \
        --lr_scheduler_type "constant" \
        --distill_loss_weight 2 \
        --do_prune \
        --output_dir "./sparse_mrpc_distilbert" \
        --target_sparsity 0.5 \
        --pruning_pattern "2:4" \
        --pruning_frequency 50