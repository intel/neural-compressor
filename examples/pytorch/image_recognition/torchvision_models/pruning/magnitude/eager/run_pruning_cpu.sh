python -u main.py \
    /path/to/imagenet/ \
    --topology resnet18 \
    --prune \
    --pretrained \
    --pruning_type magnitude \
    --initial_sparsity 0.0 \
    --target_sparsity 0.40 \
    --start_epoch 0 \
    --end_epoch 9 \
    --epochs 10 \
    --output-model saved_results \
    --batch-size 256 \
    --keep-batch-size \
    --lr 0.001

