CUDA_VISIBLE_DEVICES=2 python \
    examples/pytorch/nlp/huggingface_models/language-modeling/pruning/eager/run_clm_no_trainer.py \
    --model_name_or_path facebook/opt-1.3b \
    --dataset_name lambada \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 16 \
    --max_train_steps 3002 \
    --weight_decay  0 \
    --block_size 512 \
    --do_prune \
    --auto_slim \
    --output_dir sparse_clm_models/ \
    --target_sparsity 0.2 \
    --pruning_pattern channelx1 \
    --pruning_frequency 500 \
    --num_warmup_steps 1 \
    > sparse_logs/retrainfree_20sparsity_500fre.log 2>&1 &
