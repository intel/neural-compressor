Distillation for Quantization
============

1. [Introduction](#introduction)


2. [Distillation for Quantization Support Matrix](#distillation-for-quantization-support-matrix)


3. [Get Started with Distillation for Quantization API](#get-started-with-api)


4. [Examples](#examples)



### Introduction

Distillation and quantization are both promising methods to reduce the computational and memory footprint that huge transformer-based networks require. Quantization refers to a process of reducing the bit precision for both activations and weights. Distillation method transfers knowledge from a heavy teacher model to a light one (student) and it could be used as a performance-booster in lower-bits quantizations. Quantization-aware training recovers accuracy degradation from representation loss in the retraining process and typically provides better performance compared to post-training quantization.  
Intel provides a quantization-aware training (QAT) method that incorporates a novel layer-by-layer knowledge distillation step for INT8 quantization pipelines.



### Distillation for Quantization Support Matrix

|**Algorithm**                      |**PyTorch**   |**TensorFlow (Deprecated)** |
|---------------------------------|:--------:|:---------:|
|Distillation for Quantization    |&#10004;  |&#10006;   |



### Get Started with Distillation for Quantization API

User can pass the customized training/evaluation functions to `Distillation` for quantization tasks. In this case, distillation process can be done by pre-defined hooks in Neural Compressor. Users could place those hooks inside the quantization training function.

Neural Compressor defines several hooks for user pass

```
on_train_begin() : Hook executed before training begins
on_after_compute_loss(input, student_output, student_loss) : Hook executed after each batch inference of student model
on_epoch_end() : Hook executed at each epoch end
```

Following section illustrates how to use hooks in user pass-in training function:

```python
def training_func_for_nc(model):
    compression_manager.on_train_begin()
    for epoch in range(epochs):
        compression_manager.on_epoch_begin(epoch)
        for i, batch in enumerate(dataloader):
            compression_manager.on_step_begin(i)
            ......
            output = model(batch)
            loss = output.loss
            loss = compression_manager.on_after_compute_loss(batch, output, loss)
            loss.backward()
            compression_manager.on_before_optimizer_step()
            optimizer.step()
            compression_manager.on_step_end()
        compression_manager.on_epoch_end()
    compression_manager.on_train_end()
...
```

In this case, the launcher code is like the following:

```python
from neural_compressor.experimental import common, Distillation, Quantization
from neural_compressor.config import DistillationConfig, KnowledgeDistillationLossConfig
from neural_compressor import QuantizationAwareTrainingConfig
from neural_compressor.training import prepare_compression

combs = []
distillation_criterion = KnowledgeDistillationLossConfig()
d_conf = DistillationConfig(teacher_model=teacher_model, criterion=distillation_criterion)
combs.append(d_conf)
q_conf = QuantizationAwareTrainingConfig()
combs.append(q_conf)
compression_manager = prepare_compression(model, combs)
model = compression_manager.model

model = training_func_for_nc(model)
eval_func(model)
```

### Examples

For examples of distillation for quantization, please refer to [distillation-for-quantization examples](../../examples/pytorch/nlp/huggingface_models/text-classification/optimization_pipeline/distillation_for_quantization/fx/README.md)
