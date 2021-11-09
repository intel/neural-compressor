### Prepare an environment
```shell
pip install -r requirements.txt
```

### Prepare dataset 
Please update the data path with /path/to/imagenet in "run" scripts.

### Prepare configuration
Please update the configuration of pruner in **conf.yaml**. Configuration of training is separated from **conf.yaml**

### Run
#### Non-distributed
**run_pruning_cpu.sh** is an example.
```shell
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
```

#### Distributed 
**run_pruning_distributed_cpu.sh** is an example.
```shell
horovodrun -np 2 python -u main.py \
    /path/to/imagenet/ \
    --topology resnet18 \
    --prune \
    --config conf.yaml \
    --pretrained \
    --output-model model_final.pth \
    --world-size 1 \
    --num-per-node 2 \
    --batch-size 256 \
    --keep-batch-size \
    --lr 0.001 \
    --iteration 30 \
```

### Other notes
- Topology supports resnet18/resnet34/resnet50/resnet101
- World-size and num-per-node should match to np of horovodrun. For example as run_pruning_distributed_cpu.sh, np of horovodrun is 2, and world-size * num-per-node = 1 * 2 = 2.
