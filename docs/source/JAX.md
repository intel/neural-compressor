JAX
=====

- [Introduction](#introduction)
- [Quantization API](#quantization-api)
- [Post-Training Static Quantization](#post-training-static-quantization)
- [Examples](#examples)
- [Backend and Device](#backend-and-device)


## Introduction

`neural_compressor_jax` provides an API for applying quantization on Keras models such as ViT and Gemma3.
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
        model:          FP32 Keras model to be quantized.
        quant_config:   Quantization configuration.
        calib_function: Function used for model calibration, required for static quantization.
        inplace:        When True, the original model is modified in-place and should not be used
                        afterward. A value of False is not yet supported.

    Returns:
        The quantized model.
    """
```

## Post-Training Static Quantization

The maximum absolute values of weights and activations are collected offline using a *calibration* dataset.
This dataset should be representative of the data distribution expected during inference.
The calibration process runs on the original FP32 model and records tensor distributions for scale calculations.
Typically, preparing several dozen samples is sufficient for calibration.

## Examples

Examples of how to quantize a model and use a pre-quantized model can be found below:

- [Gemma3](../../examples/jax/keras/gemma/README.md)
- [ViT](../../examples/jax/keras/vit/README.md)
- [Simple model – quantization](../../examples/jax/keras/helloworld/example_static.py)
- [Simple model – save and load](../../examples/jax/keras/helloworld/example_saving.py)

## Backend and Device

Although Intel® Neural Compressor can run on any platform supporting 8-bit floating point with Keras using the JAX backend,
performance improvements from quantization will be visible on Intel® Xeon® processors
(with AMX-FP8 extension) with JAX version greater than [v0.9](https://github.com/jax-ml/jax/releases/tag/jax-v0.9.0)
(see the full [JAX releases](https://github.com/jax-ml/jax/releases) page).

To enable performance improvements from quantization, certain JAX/XLA features must be enabled by setting the following environment variable:

```bash
export XLA_FLAGS="\
    --xla_cpu_experimental_onednn_custom_call=true --xla_cpu_use_onednn=false \
    --xla_cpu_experimental_ynn_fusion_type=invalid --xla_cpu_use_xnnpack=false \
    --xla_backend_extra_options=xla_cpu_disable_new_fusion_emitter"
```

Without this flag, quantized model operates in fake quantization mode, where tensors are rounded to the specified FP8 format but computations are still performed in 32-bit floating-point format.
