
PyTorch Weight Only Quantization
===============

1. [Introduction](#introduction)
2. [Usage](#usage)  
   2.1 [RTN](#RTN)   
   2.2 [GPTQ](#GPTQ)   
   2.3 [AutoRound](#AutoRound)
   2.4 [AWQ](#AWQ)
   2.5 [TEQ](#TEQ)
   2.6 [HQQ](#HQQ)
3. [Examples](#examples) 

## Introduction

The INC 3x New API provides support for quantizing PyTorch models using WeightOnlyQuant, with or without accuracy-aware tuning.

For detailed information on quantization fundamentals, please refer to the Quantization document.


## Get Started


### RTN

``` python
# Quantization code
from neural_compressor.torch.quantization import prepare, convert, RTNConfig

quant_config = RTNConfig()
model = prepare(model, quant_config)
model = convert(model)
```

### GPTQ

``` python
# Quantization code
from neural_compressor.torch.quantization import prepare, convert, GPTQConfig

quant_config = GPTQConfig()
model = prepare(model, quant_config)
run_fn(model) # calibration
model = convert(model)
```

### AutoRound

``` python
# Quantization code
from neural_compressor.torch.quantization import prepare, convert, AutoRoundConfig

quant_config = AutoRoundConfig()
model = prepare(model, quant_config)
run_fn(model) # calibration
model = convert(model)
```

### AWQ

``` python
# Quantization code
from neural_compressor.torch.quantization import prepare, convert, AWQConfig

quant_config = AWQConfig()
model = prepare(model, quant_config, example_inputs=example_inputs)
run_fn(model) # calibration
model = convert(model)
```

### TEQ

``` python
# Quantization code
from neural_compressor.torch.quantization import prepare, convert, TEQConfig

quant_config = TEQConfig()
model = prepare(model, quant_config, example_inputs=example_inputs)
train_fn(model) # calibration
model = convert(model)
```

### HQQ

``` python
# Quantization code
from neural_compressor.torch.quantization import prepare, convert, HQQConfig

quant_config = HQQConfig()
model = prepare(model, quant_config)
run_fn(model) # calibration
model = convert(model)
```


## Examples

Users can also refer to [examples](https://github.com/intel/neural-compressor/blob/master/examples/3.x_api/torch/llm on how to quantize a  model with WeightOnlyQuant.