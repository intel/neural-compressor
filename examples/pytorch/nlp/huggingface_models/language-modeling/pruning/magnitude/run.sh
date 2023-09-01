export CUDA_VISIBLE_DEVICES=0
python run_clm_no_trainer_pruning.py \
    --dataset_name NeelNanda/pile-10k \
    --model_name_or_path facebook/opt-125m \
    --block_size 128 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --output_dir ./test-clm \
    --do_prune \
    --num_train_epochs 10 \
    --pruning_type "magnitude" \
    --target_sparsity 0.8 \
    --pruning_pattern "4x1" \
    --pruning_frequency 1000
