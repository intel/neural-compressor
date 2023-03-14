# Step by Step
## Channel Pruning for Consecutive Linear Layers
An interesting thing for pruning is that if we do [channel pruning](https://github.com/intel/neural-compressor/tree/master/neural_compressor/compression/pruner#pruning-patterns) for some linear layers in NLP models, we can permanently remove these all-zero channels without changing their accuracy. 

To be specific, if the model has two consecutive linear layers, which is common in both **Bert series** and **GPT series** models' FFN parts, and we conduct the input channel pruning for the second linear layer (masking weights by column). We can remove these all-zero channels. Plus, we also remove the same indices output channels in the first linear layers (masking weights by row), since their contribution for activation will be masked by the second layer's. 

This leads to no change for model's accuracy, but can obtain a significant acceleration for model's inference, because the transformer models' FFN parts take nearly 50% of entire computing overhead. Thus, compressing weights in FFN parts is really useful.

## API for Consecutive Linear Compression
We provide an API function to process your sparsity model if you have done channel pruning for linear layers described above. Here is how you call our API function. Our API integrate the function of searching consecutive linear layers automatically and compress their weights if channel sparsity is detected. 
```python
from neural_compressor.training import PruningCallbacks
# automatically slim your model. 
model = PruningCallbacks.model_slim(model)
```

## Run Examples
We provides an example of Bert-Base to demonstrate how we slim Transformer-based models. simply run the following script:
```bash
sh run_qa.sh
```
After FFN compression, the inference speed of the model will be significantly improved on both CPU and GPU.

For more information about pruning, please refer to our [INC Pruning API](https://github.com/intel/neural-compressor/tree/master/neural_compressor/compression/pruner).
