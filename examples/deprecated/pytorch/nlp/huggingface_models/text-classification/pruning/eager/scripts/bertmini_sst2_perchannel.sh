#!/bin/bash
set -x

    python3  examples/pytorch/nlp/huggingface_models/text-classification/pruning/eager/run_glue_no_trainer.py \
        --model_name_or_path "/path/to/bertmini-sst2/dense_finetuned_model" \
        --task_name "sst2" \
        --max_length 128 \
        --per_device_train_batch_size 16 \
        --learning_rate 5e-5 \
        --distill_loss_weight 2.0 \
        --num_train_epochs 15 \
        --weight_decay 5e-5   \
        --cooldown_epochs 5 \
        --sparsity_warm_epochs 1 \
        --lr_scheduler_type "constant" \
        --do_prune \
        --pruning_type "snip_momentum_progressive" \
	--output_dir "./sparse_sst2_bertmini" \
        --target_sparsity 0.6 \
        --pruning_pattern "1xchannel" \
        --pruning_frequency 500
