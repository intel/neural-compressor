# do ffn auto slim from a sparse model
# python ./run_ffn_slim_only.py \
#         --model_name_or_path "/path/to/your/sparse/model/" \
#         --dataset_name squad \
#         --max_seq_length 384 \
#         --doc_stride 128 \

# a pipeline: first do ffn pruning, then do auto_slim
python ./run_ffn_slim_pipeline.py \
        --model_name_or_path "/path/to/your/dense/model/" \
        --dataset_name squad \
        --max_seq_length 384 \
        --doc_stride 128 \
        --per_device_train_batch_size 12 \
        --do_prune \
        --num_warmup_steps 1000 \
        --weight_decay 1e-7 \
        --learning_rate 7e-5 \
        --cooldown_epoch 4 \
        --num_train_epochs 10 \
        --distill_loss_weight 4.5 \
        --pruning_frequency 1000 \
        --prune_ffn2_sparsity 0.90 \
        --auto_slim \
        --output_dir "/path/to/save/slim/model/" 
