#!/bin/bash

python3 ./examples/pytorch/nlp/huggingface_models/translation/pruning/eager/run_translation_no_trainer.py \
    --model_name_or_path '/path/to/Flan-T5/unfinetuned_model/' \
    --source_lang en \
    --target_lang ro \
    --source_prefix "translate English to Romanian: " \
    --num_warmup_steps 5000 \
    --num_train_epochs 10 \
    --dataset_name wmt16 \
    --dataset_config_name ro-en \
    --output_dir "/dense-finetuned-Flan-T5" \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --checkpointing_steps 'epoch' \
    --weight_decay 0 \
    --learning_rate 5e-04               



