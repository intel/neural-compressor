
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
    2.7 [Specify Quantization Rules](#specify-quantization-rules)   
    2.8 [Save&Load](save-and-load)
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
### Specify Quantization Rules
Intel(R) Neural Compressor support specify quantization rules by operator name or operator type. Users can set `local` in dict or use `set_local` method of config class to achieve the above purpose.

1. Example of setting `local` from a dict
```python
quant_config = {
    "rtn": {
        "global": {
            "dtype": "int",
            "bits": 4,
            "group_size": -1,
            "use_sym": True,
        },
        "local": {
            "lm_head": {
                "dtype": "fp32",
            },
        },
    }
}
```
2. Example of using `set_local`
```python
quant_config = RTNConfig()
lm_head_config = RTNConfig(dtype="fp32")
quant_config.set_local("lm_head", lm_head_config)
```

### Save and Load
The saved_results folder contains two files: quantized_model.pt and qconfig.json, and the generated q_model is a quantized model.
```python
# Quantization code
from neural_compressor.torch.quantization import prepare, convert, AutoRoundConfig

quant_config = AutoRoundConfig()
model = prepare(model, quant_config)
run_fn(model) # calibration
model = convert(model)

# save
model.save("saved_results")

# load
from neural_compressor.torch.quantization import load
orig_model = YOURMODEL()
loaded_model = load("saved_model", model=orig_model) # Please note that the model parameter passes the original model.
```


## Examples

Users can also refer to [examples](https://github.com/intel/neural-compressor/blob/master/examples/3.x_api/torch/llm on how to quantize a  model with WeightOnlyQuant.