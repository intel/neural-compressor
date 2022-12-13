python -u main.py \
    /path/to/imagenet/ \
    --topology resnet18 \
    --prune \
    --config conf.yaml \
    --pretrained \
    --output-model model_final.pth \
    --batch-size 256 \
    --keep-batch-size \
    --lr 0.001 \
    --iteration 30 \

