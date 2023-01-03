Distillation for Quantization
============

### Introduction

Distillation and quantization are both promising methods to reduce the computational and memory footprint that huge transformer-based networks require. Quantization refers to a process of reducing the bit precision for both activations and weights. Distillation method transfers knowledge from a heavy teacher model to a light one (student) and it could be used as a performance-booster in lower-bits quantizations. Quantization-aware training recovers accuracy degradation from representation loss in the retraining process and typically provides better performance compared to post-training quantization. 
Intel provides a quantization-aware training (QAT) method that incorporates a novel layer-by-layer knowledge distillation step for INT8 quantization pipelines. 

### User-defined yaml

The configurations of distillation and QAT are specified in distillation.yaml and qat.yaml, respectively.


### Examples

For examples of distillation for quantization, please refer to [distillation-for-quantization examples](../examples/pytorch/nlp/huggingface_models/text-classification/optimization_pipeline/distillation_for_quantization/fx/README.md)
