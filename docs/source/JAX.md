JAX
=====

- [Introduction](#introduction)
- [Quantization API](#quantization-api)
- [Post-Training Static Quantization](#post-training-static-quantization)
- [Examples](#examples)
- [Backend and Device](#backend-and-device)


## Introduction

`neural_compressor.jax` provides an API for applying quantization to Keras models such as ViT and Gemma3.
Since only JAX is supported as the Keras backend, the environment variable `KERAS_BACKEND` should be set to `jax`.
The following 8-bit floating-point formats are supported: `fp8_e4m3` and `fp8_e5m2`.

Quantized models can be saved and loaded using standard Keras APIs
([save_model](https://keras.io/api/models/model_saving_apis/model_saving_and_loading/#savemodel-function) and
[load_model](https://keras.io/api/models/model_saving_apis/model_saving_and_loading/#loadmodel-function))
or Keras Hub APIs
([save_to_preset](https://keras.io/keras_hub/api/base_classes/task/#savetopreset-method) and
[from_preset](https://keras.io/keras_hub/api/base_classes/task/#frompreset-method)).
This approach allows users to take advantage of pre-quantized models with minimal code change - just add one line:
```python
import neural_compressor.jax
 ```

Quantization was developed primarily to improve the performance of Keras models on Intel® Xeon® processors,
but it can potentially be used on other platforms as well.

## Quantization API

```python
def quantize_model(
    model: keras.Model,
    quant_config: BaseConfig,
    calib_function: Callable = None,
    inplace: bool = True
):
    """Return a quantized Keras model according to the given configuration.

    Args:
        model:          FP32/BF16 Keras model to be quantized.
        quant_config:   Quantization configuration.
        calib_function: Function used for model calibration, required for static quantization.
        inplace:        When True, the original model is modified in-place and should not be used
                        afterward. A value of False is not yet supported.

    Returns:
        The quantized model.
    """
```

## Quantization configs

The quantization configuration can be defined in the code, for example:
```python
from neural_compressor.jax import StaticQuantConfig
quant_config = StaticQuantConfig(weight_dtype="fp8_e4m3", activation_dtype="fp8_e4m3")
```
Quantization configurations can also be saved as json and loaded via *JaxBaseConfig.from_json_file()*
```python
from neural_compressor.jax import StaticQuantConfig, JaxBaseConfig
quant_config = StaticQuantConfig(weight_dtype="fp8_e4m3", activation_dtype="fp8_e4m3")
quant_config.to_json_file("path/to/quant_config.json")
loaded_quant_cofnig = JaxBaseConfig.from_json_file("path/to/quant_config.json")
```

### White list and exclude list
Quantization configs provide a way to include and exclude specific layers and classes via *white_list* and *exclude_list* parameters - the *exclude_list* takes priority over the *white_list*.
Available formats are:
 - layer path regex
 - layer class name (string)
 - layer class
See example below:
```python
from keras.layers import EinsumDense
from neural_compressor.jax import StaticQuantConfig
# path regex - matches paths like "model/encoder_{i}/mha", except "model/encoder_2/mha"
cfg1 = StaticQuantConfig(weight_dtype="fp8_e4m3", activation_dtype="fp8_e4m3", white_list=[".*mha"], exclude_list=[".*encoder_2.*mha"])
# class name - matches all Dense layers
cfg2 = StaticQuantConfig(weight_dtype="fp8_e4m3", activation_dtype="fp8_e4m3", white_list=["Dense"])
# class - matches all EinsumDense layers
cfg3 = StaticQuantConfig(weight_dtype="fp8_e4m3", activation_dtype="fp8_e4m3", white_list=["Einsum"])
```

### Composable configs

It's possible to combine multiple configs to achieve per-layer and per-class configuration, by combining multiple configs into a *JaxComposableConfig*:
```python
from neural_compressor.jax import StaticQuantConfig, DynamicQuantConfig
# * ``first``  -> only ``static1`` matches                -> static
# * ``second`` -> ``static1`` and ``dynamic`` match       -> dynamic (later)
# * ``third``  -> ``dynamic`` and ``static2`` match       -> static  (later)
static1 = StaticQuantConfig(weight_dtype="int8", activation_dtype="int8", white_list=["dense1", "dense2"])
dynamic = DynamicQuantConfig(weight_dtype="fp8_e4m3", activation_dtype="fp8_e4m3", white_list=["dense2", "dense3"])
# white_list everything (Dense) but exclude the layers claimed earlier -> resolves to 'third'.
static2 = StaticQuantConfig(
    weight_dtype="int8", activation_dtype="int8", white_list=["Dense"], exclude_list=["dense1", "dense2"]
)

# Compose the configurations into a single configuration
# It's also possible to construct the composed config via JaxComposableConfig(static1, dynamic, static2)
config = static1 + dynamic + static2
```
***NOTE*** 
The order in which the configs are provided matters - multiple configs that apply for the same layer are resolved in a last-wins manner, as explained above.

Config examples can be found under [configs](../../examples/jax/keras/configs/) directory.


## Post-Training Static Quantization

The maximum absolute values of weights and activations are collected offline using a *calibration* dataset.
This dataset should be representative of the data distribution expected during inference.
The calibration process runs on the original FP32/BF16 model and records tensor distributions for scale calculations.
Typically, preparing several dozen samples is sufficient for calibration.

## Examples

Examples of how to quantize a model and use a pre-quantized model can be found below:

- [Gemma3](../../examples/jax/keras/gemma/README.md)
- [ViT](../../examples/jax/keras/vit/README.md)
- [Simple model – quantization](../../examples/jax/keras/helloworld/example.py)
- [Simple model - quantization with composable config](../../examples/jax/keras/helloworld/example_composable_config.py)
- [Simple model – save and load](../../examples/jax/keras/helloworld/example_saving.py)
- [Simple model - composable configs](../../examples/jax/keras/helloworld/example_composable_configs.py)

## Backend and Device

Although Intel® Neural Compressor can run on any platform supporting 8-bit floating point with Keras using the JAX backend,
performance improvements from quantization will be visible on Intel® Xeon® processors
(with AMX-FP8 extension) with JAX version greater than [v0.9](https://github.com/jax-ml/jax/releases/tag/jax-v0.9.0)
(see the full [JAX releases](https://github.com/jax-ml/jax/releases) page).
