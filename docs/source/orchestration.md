Optimization Orchestration
============

1. [Introduction](#introduction)

    1.1. [One-shot](#one-shot)

2. [Orchestration Support Matrix](#orchestration-support-matrix)
3. [Get Started with Orchestration API ](#get-started-with-orchestration-api)
4. [Examples](#examples)

## Introduction

Orchestration is the combination of multiple optimization techniques, either applied simultaneously (one-shot). Intel Neural Compressor supports arbitrary meaningful combinations of supported optimization methods under one-shot, such as pruning during quantization-aware training.

### One-shot
Since quantization-aware training, pruning and distillation all leverage training process for optimization, we can achieve the goal of optimization through one shot training with arbitrary meaningful combinations of these methods, which often gain more benefits in terms of performance and accuracy than just one compression technique applied, and usually are as efficient as applying just one compression technique. The three possible combinations are shown below.
- Pruning during quantization-aware training
- Distillation with pattern lock pruning
- Distillation with pattern lock pruning and quantization-aware training

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
    </tbody>
</table>

## Get Started with Orchestration API 

Neural Compressor defines `Scheduler` class to automatically pipeline execute model optimization with one shot way. 

User instantiates model optimization components, such as quantization, pruning, distillation, separately. After that, user could append
those separate optimization objects into scheduler's pipeline, the scheduler API executes them one by one.

In following example it execute the distillation and pruning with one-shot way, the code is like below.

```python
from neural_compressor.training import prepare_compression
from neural_compressor.config import DistillationConfig, KnowledgeDistillationLossConfig, WeightPruningConfig
distillation_criterion = KnowledgeDistillationLossConfig()
d_conf = DistillationConfig(model, distillation_criterion)
p_conf = WeightPruningConfig()
compression_manager = prepare_compression(model=model, confs=[d_conf, p_conf])

compression_manager.callbacks.on_train_begin()
train_loop:
    compression_manager.on_train_begin()
    for epoch in range(epochs):
        compression_manager.on_epoch_begin(epoch)
        for i, batch in enumerate(dataloader):
            compression_manager.on_step_begin(i)
            ......
            output = model(batch)
            loss = ......
            loss = compression_manager.on_after_compute_loss(batch, output, loss)
            loss.backward()
            compression_manager.on_before_optimizer_step()
            optimizer.step()
            compression_manager.on_step_end()
        compression_manager.on_epoch_end()
    compression_manager.on_train_end()
    
model.save('./path/to/save')

```

## Examples

[Orchestration Examples](../../examples/README.md#orchestration)
