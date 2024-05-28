
Quantization
===============

1. [Quantization Introduction](#quantization-introduction)
2. [Quantization Fundamentals](#quantization-fundamentals)
3. [Accuracy Aware Tuning](#accuracy-aware-tuning)

4. [Get Started](#get-started)  
   5.1 [Without Accuracy Aware Tuning](#without-accuracy-aware-tuning)   
   5.2 [With Accuracy Aware Tuning](#with-accuracy-aware-tuning)   
   5.3 [Specify Quantization Rules](#specify-quantization-rules)  

## Quantization Introduction

Quantization is a very popular deep learning model optimization technique invented for improving the speed of inference. It minimizes the number of bits required by converting a set of real-valued numbers into the lower bit data representation, such as int8 and int4, mainly on inference phase with minimal to no loss in accuracy. This way reduces the memory requirement, cache miss rate, and computational cost of using neural networks and finally achieve the goal of higher inference performance. On Intel 3rd Gen Intel® Xeon® Scalable Processors, user could expect up to 4x theoretical performance speedup. We expect further performance improvement with [Intel® Advanced Matrix Extensions](https://www.intel.com/content/www/us/en/products/docs/accelerator-engines/advanced-matrix-extensions/overview.html) on 4th Gen Intel® Xeon® Scalable Processors.

## Quantization Fundamentals

`Affine quantization` and `Scale quantization` are two common range mapping techniques used in tensor conversion between different data types. 

For TensorFlow, all quantizable operators support `Scale quantization`, while a parts of operators support `Affine quantization`. For Keras, the quantizable layers only support `Scale quantization`.

The math equation is like: $$X_{int8} = round(Scale \times X_{fp32} + ZeroPoint)$$.

**Affine Quantization**

This is so-called `asymmetric quantization`, in which we map the min/max range in the float tensor to the integer range. Here int8 range is [-128, 127], uint8 range is [0, 255]. 

here:

If INT8 is specified, $Scale = (|X_{f_{max}} - X_{f_{min}}|) / 127$ and $ZeroPoint = -128 - X_{f_{min}} / Scale$.

or

If UINT8 is specified, $Scale = (|X_{f_{max}} - X_{f_{min}}|) / 255$ and $ZeroPoint = - X_{f_{min}} / Scale$.

**Scale Quantization**

This is so-called `Symmetric quantization`, in which we use the maximum absolute value in the float tensor as float range and map to the corresponding integer range. 

The math equation is like:

here:

If INT8 is specified, $Scale = max(abs(X_{f_{max}}), abs(X_{f_{min}})) / 127$ and $ZeroPoint = 0$. 

or

If UINT8 is specified, $Scale = max(abs(X_{f_{max}}), abs(X_{f_{min}})) / 255$ and $ZeroPoint = 128$.


> ***Note***
> Sometimes the reduce_range feature, that's using 7 bit width (1 sign bit + 6 data bits) to represent int8 range, may be needed on some early Xeon platforms, it's because those platforms may have overflow issues due to fp16 intermediate calculation result when executing int8 dot product operation. After AVX512_VNNI instruction is introduced, this issue gets solved by supporting fp32 intermediate data.



### Quantization Approaches

Quantization has three different approaches:
1) post training dynamic quantization
2) post training static quantization
3) quantization aware training.

Currently, only `post training static quantization` is supported by INC TF 3X API. For this approach, the min/max range in weights and activations are collected offline on a so-called `calibration` dataset. This dataset should be able to represent the data distribution of those unseen inference dataset. The `calibration` process runs on the original fp32 model and dumps out all the tensor distributions for `Scale` and `ZeroPoint` calculations. Usually preparing 100 samples are enough for calibration.

This approach is major quantization approach people should try because it could provide the better performance comparing with `post training dynamic quantization`.


## Accuracy Aware Tuning

Accuracy aware tuning is one of unique features provided by Intel(R) Neural Compressor, compared with other 3rd party model compression tools. This feature can be used to solve accuracy loss pain points brought by applying low precision quantization and other lossy optimization methods. 

This tuning algorithm creates a tuning space by querying framework quantization capability and model structure, selects the ops to be quantized by the tuning strategy, generates quantized graph, and evaluates the accuracy of this quantized graph. The optimal model will be yielded if the pre-defined accuracy goal is met.

Neural compressor also support to quantize all quantizable ops without accuracy tuning, user can decide whether to tune the model accuracy or not. Please refer to "Get Start" below.

### Working Flow

User could refer to below chart to understand the whole tuning flow.

<img src="../../source/imgs/accuracy_aware_tuning_flow.png" width=600 height=480 alt="accuracy aware tuning working flow">


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

This means user could leverage the advance feature of Intel(R) Neural Compressor to tune out a best quantized model which has best accuracy and good performance. User should provide either `eval_fn` and `eval_args`.

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
