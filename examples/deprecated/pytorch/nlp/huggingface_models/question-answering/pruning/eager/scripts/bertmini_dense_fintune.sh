#!/bin/bash
set -x

    python examples/pytorch/nlp/huggingface_models/question-answering/pruning/eager/run_qa_no_trainer.py \
    --model_name_or_path "prajjwal1/bert-mini" \
    --dataset_name "squad" \
    --max_seq_length 384 \
    --doc_stride 128 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --num_warmup_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 5 \
    --output_dir "./dense_qa_bertmini"