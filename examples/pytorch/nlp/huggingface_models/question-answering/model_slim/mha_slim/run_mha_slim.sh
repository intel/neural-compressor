export CUDA_VISIBLE_DEVICES=6

# only do head pruning from a sparse model
# python examples/pytorch/nlp/huggingface_models/question-answering/model_slim/mha_slim/run_mha_slim_only.py \
#         --model_name_or_path "/data1/cyy/models/useful_models/bert-base-local-head70-8870/" \
#         --dataset_name squad \
#         --max_seq_length 384 \
#         --doc_stride 128 \

# a pipeline: first do pruning, and then do auto slim
/home/cyy/anaconda3/envs/cyy_pruning/bin/python examples/pytorch/nlp/huggingface_models/question-answering/model_slim/mha_slim/run_mha_slim_pipeline.py \
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
        --prune_mha_sparsity 0.70 \
        --auto_slim \
        --output_dir "./bert-base-mha70-v2"