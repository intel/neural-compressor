
TensorFlow Quantization
===============

1. [Introduction](#introduction)
2. [Get Started](#get-started)  
   2.1 [Without Accuracy Aware Tuning](#without-accuracy-aware-tuning)   
   2.2 [With Accuracy Aware Tuning](#with-accuracy-aware-tuning)   
   2.3 [Specify Quantization Rules](#specify-quantization-rules) 
3. [Examples](#examples) 

## Introduction

`neural_compressor.tensorflow` supports quantizing both TensorFlow and Keras model with or without accuracy aware tuning.

For the detailed quantization fundamentals, please refer to the document for [Quantization](quantization.md).

## Get Started

### Without Accuracy Aware Tuning

This means user could leverage Intel(R) Neural Compressor to directly generate a fully quantized model without accuracy aware tuning. It's user responsibility to ensure the accuracy of the quantized model meets expectation.

``` python
# main.py

# Original code
model = tf.keras.applications.resnet50.ResNet50(weights="imagenet")
val_dataset = ...
val_dataloader = MyDataloader(dataset=val_dataset)

# Quantization code
from neural_compressor.tensorflow import quantize_model, StaticQuantConfig

quant_config = StaticQuantConfig()
qmodel = quantize_model(
    model=model,
    quant_config=quant_config,
    calib_dataloader=val_dataloader,
)
qmodel.save("./output")
```

### With Accuracy Aware Tuning

This means user could leverage the advance feature of Intel(R) Neural Compressor to tune out a best quantized model which has best accuracy and good performance. User should provide `eval_fn` and `eval_args`.

``` python
# main.py

# Original code
model = tf.keras.applications.resnet50.ResNet50(weights="imagenet")
val_dataset = ...
val_dataloader = MyDataloader(dataset=val_dataset)


def eval_acc_fn(model) -> float:
    ...
    return acc


# Quantization code
from neural_compressor.common.base_tuning import TuningConfig
from neural_compressor.tensorflow import autotune

# it's also supported to define custom_tune_config as:
# TuningConfig(StaticQuantConfig(weight_sym=[True, False], act_sym=[True, False]))
custom_tune_config = TuningConfig(
    config_set=[
        StaticQuantConfig(weight_sym=True, act_sym=True),
        StaticQuantConfig(weight_sym=False, act_sym=False),
    ]
)
best_model = autotune(
    model=model,
    tune_config=custom_tune_config,
    eval_fn=eval_acc_fn,
    calib_dataloader=val_dataloader,
)
best_model.save("./output")
```

### Specify Quantization Rules
Intel(R) Neural Compressor support specify quantization rules by operator name or operator type. Users can set `local` in dict or use `set_local` method of config class to achieve the above purpose.

1. Example of setting `local` from a dict
```python
quant_config = {
    "static_quant": {
        "global": {
            "weight_dtype": "int8",
            "weight_sym": True,
            "weight_granularity": "per_tensor",
            "act_dtype": "int8",
            "act_sym": True,
            "act_granularity": "per_tensor",
        },
        "local": {
            "conv1": {
                "weight_dtype": "fp32",
                "act_dtype": "fp32",
            }
        },
    }
}
config = StaticQuantConfig.from_dict(quant_config)
```
2. Example of using `set_local`
```python
quant_config = StaticQuantConfig()
conv2d_config = StaticQuantConfig(
    weight_dtype="fp32",
    act_dtype="fp32",
)
quant_config.set_local("conv1", conv2d_config)
```

## Examples

Users can also refer to [examples](https://github.com/intel/neural-compressor/blob/master/examples/tensorflow) on how to quantize a TensorFlow model with `neural_compressor.tensorflow`.
