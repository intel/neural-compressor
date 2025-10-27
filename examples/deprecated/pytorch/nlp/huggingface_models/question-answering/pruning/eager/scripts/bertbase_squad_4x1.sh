#!/bin/bash
set -x

    python examples/pytorch/nlp/huggingface_models/question-answering/pruning/eager/run_qa_no_trainer.py \
        --model_name_or_path "/path/to/bert-base-uncased-squad/dense_finetuned_model/" \
        --dataset_name squad \
        --max_seq_length 384 \
        --doc_stride 128 \
        --per_device_train_batch_size 12 \
        --do_prune \
        --num_warmup_steps 1000 \
        --output_dir "./sparse_qa_bertbase" \
        --weight_decay 1e-7 \
        --learning_rate 7e-5 \
        --cooldown_epoch 4 \
        --num_train_epochs 10 \
        --distill_loss_weight 4.5 \
        --target_sparsity 0.8 \
        --pruning_pattern "4x1" \
        --pruning_frequency 1000