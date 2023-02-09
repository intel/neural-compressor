#!/bin/bash
set -x

    python examples/pytorch/nlp/huggingface_models/text-classification/pruning/eager/run_glue_no_trainer.py \
         --model_name_or_path "/path/to/bertmini/pretrained_model/" \
        --task_name "sst2" \
        --max_length 128 \
        --per_device_train_batch_size 32 \
        --learning_rate 5e-5 \
        --num_train_epochs 10 \
        --output_dir "./dense_sst2_bertmini"