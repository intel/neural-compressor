# Step by Step

## Head Pruning for Self-Attention Layers
Self attention modules are common in all Transformer-based models. These models use multi-head attention (also known as MHA) to enhance their abilities of linking contextual information. Transformer-based models usually stack a sequence of MHA modules, and this makes MHA takes a noticable storage and memory bandwith. As an optimization method, head pruning removes attention heads which make minor contribution to model's contextual analysis. This method does not lead to much accuracy loss, but provides us with much opportunity for model acceleration. 

## API for MHA Head Pruning
We provide API functions for you to complete the process above and slim your transformer models easily. Here is how to call our API functions. Simply provide a target sparsity value to our Our API function **parse_auto_slim_config** and it can generate the [pruning_configs](https://github.com/intel/neural-compressor/tree/master/neural_compressor/compression/pruner#get-started-with-pruning-api) used by our pruning API. Such process is fully automatic and target multi-head attention layers will be included without manual setting. After pruning process finished, use API function **model_slim** to slim the model.

```python
# auto slim config
# part1 generate pruning configs for the second linear layers. 
pruning_configs = []
from neural_compressor.compression import parse_auto_slim_config
auto_slim_configs = parse_auto_slim_config(
    model, 
    mha_sparsity = prune_mha_sparsity, 
)
pruning_configs += auto_slim_configs

################
"""
# Training codes.
......
"""
################

from neural_compressor.compression import model_slim
model = model_slim(model)
```

## Run Examples
We provides an example of Bert-Base to demonstrate how we do head pruning in Transformer-based models. simply run the following script:
```bash
sh run_mha_slim_pipeline.sh
```
For more information about pruning, please refer to our [INC Pruning API](https://github.com/intel/neural-compressor/tree/master/neural_compressor/compression/pruner).
