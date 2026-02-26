PyTorch Static Quantization
========================================
1. [Introduction](#introduction)
2. [Get Started](#get-started) \
    2.1 [Static Quantization with IPEX Backend](#static-quantization-with-ipex-backend) \
        2.1.1 [Usage Sample with IPEX](#usage-sample-with-ipex) \
        2.1.2 [Specify Quantization Rules](#specify-quantization-rules) \
        2.1.3 [Model Examples](#model-examples) \
    2.2 [Static Quantization with PT2E Backend](#static-quantization-with-pt2e-backend) \
        2.2.1 [Usage Sample with PT2E](#usage-sample-with-pt2e)  
        2.2.2 [Model Examples with PT2E](#model-examples-with-pt2e)


## Introduction

Post-Training Quantization (PTQ) is a technique used to convert a pre-trained floating-point model to a quantized model. This approach does not require model retraining. Instead, it uses calibration data to determine the optimal quantization parameters. Static quantization involves calibrating both weights and activations during the quantization process. Currently, we support two paths to perform static PTQ [Intel Extension for PyTorch (IPEX)](https://github.com/intel/intel-extension-for-pytorch) and [PyTorch 2 Export Quantization (PT2E)](https://pytorch.org/tutorials/prototype/pt2e_quant_x86_inductor.html).

## Get Started

### Static Quantization with IPEX Backend

Intel Extension for PyTorch (IPEX) provides optimizations specifically for Intel hardware, improving the performance of PyTorch models through efficient execution on CPUs. IPEX supports PTQ, allowing users to quantize models to lower precision to reduce model size and inference time while maintaining accuracy.

The design philosophy of the quantization interface of Intel(R) Neural Compressor is easy-of-use. It requests user to provide `model`, `calibration function`, and `example inputs`. Those parameters would be used to quantize and tune the model. 

`model` is the framework model location or the framework model object.

`calibration function` is used to determine the appropriate quantization parameters, such as `scale` and `zero-point`, for the model's weights and activations. This process is crucial for minimizing the loss of accuracy that can occur when converting from floating-point to lower-precision format.

IPEX leverages just-in-time (JIT) compilation techniques for optimizing the model. `example inputs` is used to trace the computational graph of the model, enabling various optimizations and transformations that are specific to IPEX. This tracing process captures the operations performed by the model, allowing IPEX to apply quantization optimizations effectively. `example inputs` should be representative of the actual data the model will process to ensure accurate calibration.


#### Usage Sample with IPEX
```python
import intel_extension_for_pytorch as ipex
from neural_compressor.torch.quantization import StaticQuantConfig, convert, prepare

quant_config = StaticQuantConfig(act_sym=True, act_algo="minmax")
prepared_model = prepare(model, quant_config=quant_config, example_inputs=example_inputs)
run_fn(prepared_model)
q_model = convert(prepared_model)
```

> [!IMPORTANT]  
> To use static quantization with the IPEX backend, please explicitly import IPEX at the beginning of your program.

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
quant_config.set_local("Linear", StaticQuantConfig(w_dtype="fp32", act_dtype="fp32"))
prepared_model = prepare(model, quant_config=quant_config, example_inputs=example_inputs)
run_fn(prepared_model)
q_model = convert(prepared_model)
```

#### Model Examples

Users could refer to [examples](https://github.com/intel/neural-compressor/blob/master/examples/pytorch/nlp/huggingface_models/language-modeling/quantization/static_quant/ipex) on how to quantize a new model.


### Static Quantization with PT2E Backend
Compared to the IPEX backend, which uses JIT compilation to capture the eager model, the PT2E path uses `torch.dynamo` to capture the eager model into an FX graph model, and then inserts the observers and Q/QD pairs on it. Finally it uses the `torch.compile` to perform the pattern matching and replace the  Q/DQ pairs with optimized quantized operators.

#### Usage Sample with PT2E
There are four steps to perform W8A8 static quantization with PT2E backend: `export`, `prepare`, `convert` and `compile`.

```python
import torch
from neural_compressor.torch.export import export
from neural_compressor.torch.quantization import StaticQuantConfig, prepare, convert

# Prepare the float model and example inputs for export model
model = UserFloatModel()
example_inputs = ...

# Export eager model into FX graph model
exported_model = export(model=model, example_inputs=example_inputs)
# Quantize the model
quant_config = StaticQuantConfig()
prepared_model = prepare(exported_model, quant_config=quant_config)
# Calibrate
run_fn(prepared_model)
q_model = convert(prepared_model)
# Compile the quantized model and replace the Q/DQ pattern with Q-operator
from torch._inductor import config

config.freezing = True
opt_model = torch.compile(q_model)
```

> Note: The `set_local` of `StaticQuantConfig` will be supported after the torch 2.4 release.

#### Model Examples with PT2E

Users could refer to [cv examples](https://github.com/intel/neural-compressor/blob/master/examples/pytorch/cv/static_quant) and [llm examples](https://github.com/intel/neural-compressor/blob/master/examples/pytorch/nlp/huggingface_models/language-modeling/quantization/static_quant/pt2e) on how to quantize a new model.
