#!/bin/bash
python run_qa_no_trainer_auto_slim.py \
        --model_name_or_path "/path/to/your/bert-large-model/" \
        --dataset_name squad \
        --max_seq_length 384 \
        --doc_stride 128 \
        --per_device_train_batch_size 24 \
	--per_device_eval_batch_size 24 \
        --do_prune \
        --num_warmup_steps 1000 \
        --weight_decay 0 \
        --learning_rate 5e-5 \
        --cooldown_epoch 4 \
        --num_train_epochs 10 \
        --distill_loss_weight 3 \
        --pruning_frequency 1000 \
        --prune_ffn2_sparsity 0.80 \
        --prune_mha_sparsity 0.70 \
        --auto_slim 
