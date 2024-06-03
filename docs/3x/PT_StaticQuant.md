PyTorch Static Quantization
========================================

1. [Introduction](#introduction)
2. [Get Started](#get-started) \
    2.1 [Static Quantization with IPEX backend](#static-quantization-with-ipex-backend) \
        2.1.1 [Usage Sample](#usage-sample) \
        2.1.2 [Specify Quantization Rules](#specify-quantization-rules) \
        2.1.3 [Model Examples](#model-examples)

## Introduction

Quantization has three different approaches:
1) post training dynamic quantization
2) post training static quantization
3) quantization aware training

The first two approaches belong to optimization on inference. The last belongs to optimization during training.

Post-Training Quantization (PTQ) is a technique used to convert a pre-trained floating-point model to a quantized model. This approach does not require model retraining. Instead, it uses calibration data to determine the optimal quantization parameters. Static quantization involves calibrating both weights and activations during the quantization process.

Compared with `post training dynamic quantization`, the min/max range in weights and activations of `post training static quantization` are collected offline on calibration dataset. This dataset should be able to represent the data distribution of those unseen inference dataset. The `calibration` process runs on the original fp32 model and dumps out all the tensor distributions for `Scale` and `ZeroPoint` calculations. Usually preparing 100 samples are enough for calibration.

## Get Started

### Static Quantization with IPEX backend

Intel Extension for PyTorch (IPEX) provides optimizations specifically for Intel hardware, improving the performance of PyTorch models through efficient execution on CPUs. IPEX supports PTQ, allowing users to quantize models to lower precision to reduce model size and inference time while maintaining accuracy.

The design philosophy of the quantization interface of Intel(R) Neural Compressor is easy-of-use. It requests user to provide `model`, `calibration function`, and `example inputs`. Those parameters would be used to quantize and tune the model. 

`model` is the framework model location or the framework model object.

`calibration function` is used to determine the appropriate quantization parameters, such as `scale` and `zero-point`, for the model's weights and activations. This process is crucial for minimizing the loss of accuracy that can occur when converting from floating-point to lower-precision format.

IPEX leverages just-in-time (JIT) compilation techniques for optimizing the model. `example inputs` is used to trace the computational graph of the model, enabling various optimizations and transformations that are specific to IPEX. This tracing process captures the operations performed by the model, allowing IPEX to apply quantization optimizations effectively. `example inputs` should be representative of the actual data the model will process to ensure accurate calibration.


#### Usage Sample
```python
from neural_compressor.torch.quantization import StaticQuantConfig, convert, prepare

quant_config = StaticQuantConfig(act_sym=True, act_algo="minmax")
prepared_model = prepare(model, quant_config=quant_config, example_inputs=example_inputs)
run_fn(prepared_model)
q_model = convert(prepared_model)
```

#### Specify Quantization Rules
Intel(R) Neural Compressor support specify quantization rules by operator name or operator type. Users can use `set_local` to fallback either `op_name` or `op_type` in `StaticQuantConfig` to achieve the above purpose.

1. Example of `op_name_dict`
Here we don't quantize the layer named `fc1`.
```python
# fallback by op_name
quant_config.set_local("fc1", StaticQuantConfig(w_dtype="fp32", act_dtype="fp32"))
prepared_model = prepare(fp32_model, quant_config=quant_config, example_inputs=example_inputs)
run_fn(prepared_model)
q_model = convert(prepared_model)
```
2. Example of `op_type_dict`
Here we don't quantize `Linear` layers.
```python
# fallback by op_type
quant_config.set_local(torch.nn.Linear, StaticQuantConfig(w_dtype="fp32", act_dtype="fp32"))
prepared_model = prepare(model, quant_config=quant_config, example_inputs=example_inputs)
run_fn(prepared_model)
q_model = convert(prepared_model)
```

#### Model Examples

Users could refer to [examples](https://github.com/intel/neural-compressor/blob/master/examples/3.x_api/pytorch/nlp/huggingface_models/language-modeling/quantization/llm) on how to quantize a new model.
