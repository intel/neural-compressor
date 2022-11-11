Optimization Orchestration
============

1. [Introduction](#introduction)

    1.1. [One-shot](#one-shot)

    1.2. [Multi-shot](#multi-shot)

2. [Orchestration Support Matrix](#orchestration-support-matrix)
3. [Get Started with Orchestration API ](#get-started-with-orchestration-api)
4. [Examples](#examples)

## Introduction

Orchestration is the combination of multiple optimization techniques, either applied simultaneously (one-shot) or sequentially (multi-shot). Intel Neural Compressor supports arbitrary meaningful combinations of supported optimization methods under one-shot or multi-shot, such as pruning during quantization-aware training, or pruning and then post-training quantization, pruning and then distillation and then quantization.

### One-shot
Since quantization-aware training, pruning and distillation all leverage training process for optimization, we can achieve the goal of optimization through one shot training with arbitrary meaningful combinations of these methods, which often gain more benefits in terms of performance and accuracy than just one compression technique applied, and usually are as efficient as applying just one compression technique. The three possible combinations are shown below.
- Pruning during quantization-aware training
- Distillation with pattern lock pruning
- Distillation with pattern lock pruning and quantization-aware training
 
### Multi-shot
Of course, besides one-shot, we also support separate execution of each optimization process.
- Pruning and then post-training quantization
- Distillation and then post-training quantization
- Distillation, then pruning and post-training quantization

## Orchestration Support Matrix
<table>
    <thead>
        <tr>
            <th>Orchestration</th>
            <th>Combinations</th>
            <th>Supported</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=4>One-shot</td>
            <td>Pruning + Quantization Aware Training</td>
            <td>&#10004;</td>
        </tr>
        <tr>
            <td>Distillation + Quantization Aware Training</td>
            <td>&#10004;</td>
        </tr>
        <tr>
            <td>Distillation + Pruning</td>
            <td>&#10004;</td>
        </tr>
        <tr>
            <td>Distillation + Pruning + Quantization Aware Training</td>
            <td>&#10004;</td>
        </tr>
        <tr>
            <td rowspan=4>Multi-shot</td>
            <td>Pruning then Quantization</td>
            <td>&#10004;</td>
        </tr>
        <tr>
            <td>Distillation then Quantization</td>
            <td>&#10004;</td>
        </tr>
        <tr>
            <td>Distillation then Pruning</td>
            <td>&#10004;</td>
        </tr>
        <tr>
            <td>Distillation then Pruning then Quantization</td>
            <td>&#10004;</td>
        </tr>
    </tbody>
</table>

## Get Started with Orchestration API 

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

## Examples

[Orchestration Examples](../examples/README.md#orchestration)
