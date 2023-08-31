export CUDA_VISIBLE_DEVICES=5,6
# python run_clm_no_trainer_pruning.py \
# python -m torch.distributed.launch \
#    --nproc_per_node=1 \
#    run_clm_no_trainer_pruning.py \

accelerate launch --deepspeed_config_file ds_config.json --mixed_precision fp16 \
    run_clm_no_trainer_deepspeed.py \
    --dataset_name ./pile-10k \
    --model_name_or_path /models/opt-125m/ \
    --block_size 128 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --output_dir /tmp/test-clm \
    --num_train_epochs 1 \
    --max_train_steps 100 \
    --do_prune \
    --num_train_epochs 10 \
    --target_sparsity 0.8 \
    --pruning_pattern "4x1" \
    --pruning_frequency 1000
