#!/bin/bash

python3 ./examples/pytorch/nlp/huggingface_models/translation/pruning/eager/run_translation_no_trainer.py \
    --model_name_or_path '/path/to/Flan-T5/dense_finetuned_model/' \
    --do_prune \
    --source_lang en \
    --target_lang ro \
    --source_prefix "translate English to Romanian: " \
    --num_warmup_steps 5000 \
    --num_train_epochs 200 \
    --cooldown_epochs 100 \
    --dataset_name wmt16 \
    --dataset_config_name ro-en \
    --output_dir "/sparse-Flan-T5" \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --checkpointing_steps 'epoch' \
    --weight_decay 5e-04 \
    --target_sparsity 0.8 \
    --pruning_pattern "4x1" \
    --pruning_frequency 50000 \
    --distill_loss_weight 5.0 \
    --learning_rate 1e-03               



