#!/bin/bash
set -x

    python3 examples/pytorch/nlp/huggingface_models/text-classification/pruning/eager/run_glue_no_trainer.py \
        --model_name_or_path "/path/to/bertmini/pretrained_model/"  \
        --task_name "mrpc" \
        --max_length 128 \
        --per_device_train_batch_size 16 \
        --learning_rate 5e-5 \
        --num_train_epoch 5 \
        --weight_decay 5e-5 \
        --output_dir "./dense_mrpc_bertmini"

    