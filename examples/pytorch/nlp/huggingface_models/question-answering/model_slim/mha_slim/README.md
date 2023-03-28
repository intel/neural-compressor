# Step by Step
## Head Pruning for Self-Attention Layers
Self attention modules are common in all Transformer-based models. These models use multi-head attention (also known as MHA) to enhance their abilities of linking contextual information. Transformer-based models usually stack a sequence of MHA modules, and this makes MHA takes a noticable storage and memory bandwith. As an optimization method, head pruning removes attention heads which make minor contribution to model's contextual analysis. This method does not lead to much accuracy loss, but provides us with much opportunity for model acceleration. 

## API for MHA Head Pruning
We provide an API designed for head pruning. The API can automatically search MHA modules (indluding query, key, value layers and their subsequent feedword layers) in your models. pruning structured pattern for attention heads will also be calculated at the same time. Add layers and patterns to initialize your pruning configs and start head pruning with our [universal pruning API](https://github.com/intel/neural-compressor/tree/master/neural_compressor/compression/pruner).
```python
from neural_compressor.compression.pruner.model_slim.pattern_analyzer import SelfMHASearcher
searcher = SelfMHASearcher(model)
qkv_pattern, ffn_pattern = searcher.get_head_pattern()
qkv_layers, ffn_layers = searcher.search()
mha_pruning_config = [
    {
        "op_names": qkv_layers,
        "pattern": qkv_pattern,
        "target_sparsity": args.prune_heads,
    },
    {
        "op_names": ffn_layers,
        "pattern": ffn_pattern,
        "target_sparsity": args.prune_heads,
    }
]
pruning_configs += mha_pruning_config
configs = WeightPruningConfig(
   pruning_configs,
   ...
)
```

## Run Examples
We provides an example of Bert-Base to demonstrate how we do head pruning in Transformer-based models. simply run the following script:
```bash
sh run_qa.sh
```
For more information about pruning, please refer to our [INC Pruning API](https://github.com/intel/neural-compressor/tree/master/neural_compressor/compression/pruner).
