export CUDA_VISIBLE_DEVICES=1
# do ffn auto slim from a sparse model
# /home/cyy/anaconda3/envs/cyy_pruning/bin/python examples/pytorch/nlp/huggingface_models/question-answering/model_slim/ffn_slim/run_ffn_slim_only.py \
#         --model_name_or_path "/data1/cyy/models/useful_models/perchannel_70ratio_local_channelx1_output_eph12" \
#         --dataset_name squad \
#         --max_seq_length 384 \
#         --doc_stride 128 \

# a pipeline: first do ffn pruning, then do auto_slim
/home/cyy/anaconda3/envs/cyy_pruning/bin/python examples/pytorch/nlp/huggingface_models/question-answering/model_slim/ffn_slim/run_ffn_slim_pipeline.py \
        --model_name_or_path "/home/cyy/dense_models/baseline_squad_f188.59/" \
        --dataset_name squad \
        --max_seq_length 384 \
        --doc_stride 128 \
        --per_device_train_batch_size 12 \
        --do_prune \
        --num_warmup_steps 1000 \
        --weight_decay 1e-7 \
        --learning_rate 7e-5 \
        --cooldown_epoch 2 \
        --num_train_epochs 3 \
        --distill_loss_weight 4.5 \
        --pruning_frequency 1000 \
        --prune_ffn2_sparsity 0.90 \
        --auto_slim \
        --output_dir "./bert-base-ffn90-v1" 