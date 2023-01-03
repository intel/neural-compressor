Optimization Orchestration
============

## Introduction

Intel Neural Compressor supports arbitrary meaningful combinations of supported optimization methods under one-shot or multi-shot, such as pruning during quantization-aware training, or pruning and then post-training quantization,
pruning and then distillation and then quantization.

## Validated Orchestration Types

### One-shot
Since quantization-aware training, pruning and distillation all have training processes, we can achieve the goal of optimization through one shot training.
- Pruning during quantization-aware training
- Distillation with pattern lock pruning
- Distillation with pattern lock pruning and quantization-aware training
 
### Multi-shot
Of course, besides one-shot, we also support separate execution of each optimization process.
- Pruning and then post-training quantization
- Distillation and then post-training quantization
- Distillation, then pruning and post-training quantization

## Orchestration user facing API

Neural Compressor defines `Scheduler` class to automatically pipeline execute model optimization with one shot or multiple shots way. 

User instantiates model optimization components, such as quantization, pruning, distillation, separately. After that, user could append
those separate optimization objects into scheduler's pipeline, the scheduler API executes them one by one.

In following example it executes the pruning and then post-training quantization with two-shot way.

```python
from neural_compressor.experimental import Quantization, Pruning, Scheduler
prune = Pruning(prune_conf)
quantizer = Quantization(post_training_quantization_conf)
scheduler = Scheduler()
scheduler.model = model
scheduler.append(prune)
scheduler.append(quantizer)
opt_model = scheduler.fit()
```

If user wants to execute the pruning and quantization-aware training with one-shot way, the code is like below.

```python
from neural_compressor.experimental import Quantization, Pruning, Scheduler
prune = Pruning(prune_conf)
quantizer = Quantization(quantization_aware_training_conf)
scheduler = Scheduler()
scheduler.model = model
combination = scheduler.combine(prune, quantizer)
scheduler.append(combination)
opt_model = scheduler.fit()
```

### Examples

For orchestration one-shot related examples, please refer to [One-shot examples](../examples/pytorch/nlp/huggingface_models/question-answering/optimization_pipeline/prune_once_for_all/fx/README.md).

For orchestration multi-shot related examples, please refer to [Multi-shot examples](../examples/pytorch/image_recognition/torchvision_models/optimization_pipeline/).

### Publications
All the experiments from [Prune Once for ALL](https://arxiv.org/abs/2111.05754) can be reproduced using [Optimum-Intel](https://github.com/huggingface/optimum-intel) with Intel Neural Compressor.
