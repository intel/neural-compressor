Dynamic Quantization
===============

1. [Introduction](#introduction)
2. [Getting Started with Dynamic Quantization](#Getting-Started-with-Dynamic-Quantization)
3. [Examples](#examples)


## Introduction
Quantization is the process of converting floating point weights and activations to lower bitwidth tensors by multiplying the floating point values by a scale factor and rounding the results to whole numbers. Dynamic quantization determines the scale factor for activations dynamically based on the data range observed at runtime. We support W8A8 (quantizing weights and activations into 8 bits) dynamic quantization by leveraging torch's [`X86InductorQuantizer`](https://pytorch.org/tutorials/prototype/pt2e_quant_x86_inductor.html?highlight=x86inductorquantizer).


## Getting Started with Dynamic Quantization
There are four steps to perform W8A8 dynamic quantization: `export`, `prepare`, `convert` and `compile`.

```python
import torch
from neural_compressor.torch.export import export
from neural_compressor.torch.quantization import DynamicQuantConfig, prepare, convert

# Prepare the float model and example inputs for export model
model = UserFloatModel()
example_inputs = ...

# Export eager model into FX graph model
exported_model = export(model=model, example_inputs=example_inputs)
# Quantize the model
quant_config = DynamicQuantConfig()
prepared_model = prepare(exported_model, quant_config=quant_config)
q_model = convert(prepared_model)
# Compile the quantized model and replace the Q/DQ pattern with Q-operator
from torch._inductor import config

config.freezing = True
opt_model = torch.compile(q_model)
```

> Note: The `set_local` of `DynamicQuantConfig` will be supported after the torch 2.4 release.


## Examples
Example will be added later.
