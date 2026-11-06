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
    quant_config: JaxBaseConfig,
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

Complete usage example: [helloworld.py](../../examples/jax/keras/helloworld.py)

## Post-Training Static Quantization

The maximum absolute values of weights and activations are collected offline using a *calibration* dataset.
This dataset should be representative of the data distribution expected during inference.
The calibration process runs on the original FP32/BF16 model and records tensor distributions for scale calculations.
Typically, preparing several dozen samples is sufficient for calibration.

### SmoothQuant

[SmoothQuant](https://arxiv.org/abs/2211.10438) moves activation outliers into the weights before static quantization,
which makes per-tensor activation scales more accurate. Enable it on `StaticQuantConfig`:

```python
config = StaticQuantConfig(
    weight_dtype="fp8_e4m3",
    activation_dtype="fp8_e4m3",
    smooth_quant=True,  # default False
    smooth_quant_alpha=0.5,  # migration strength in [0, 1]
)
qmodel = quantize_model(model, config, calib_function)
```

or in a JSON config: `{"quantization_type": "static_quant", "config": {..., "smooth_quant": true, "smooth_quant_alpha": 0.5}}`.

The calibration function runs twice: first to collect per-input-channel activation ranges, then for the usual static
calibration on the smoothed model. For each group of linear layers that read the same input, the factors
`s = max|x|^alpha / max|W|^(1 - alpha)` (clipped to [0.01, 100]) multiply the weight input channels, and `1 / s` is
folded into the layer that produces the input. No operation is added, so the quantized model has the same layers,
kernels and speed as without SmoothQuant. The smoothing is part of the saved weights and the SmoothQuant options
are not stored with the model, so it loads like any other quantized model (also with INC versions without
SmoothQuant).

Supported blocks (`SMOOTH_RULES` in `neural_compressor/jax/algorithms/smooth_quant.py`):

| Block | Smoothed inputs | Folded into |
|---|---|---|
| keras-hub `Gemma3DecoderBlock` | q/k/v projections | `pre_attention_norm` |
| | gate / up projections | `pre_ffw_norm` |
| | down projection | up projection output channels |
| | attention output projection | value projection output channels (shared by the query heads of a kv head) |

Other models are left unchanged (a warning is logged).

Example: Gemma3-4B-instruct, fp8 e4m3 static quantization with per-tensor weight scales and 171 BoolQ calibration
samples, BoolQ accuracy (3270 samples) on Intel Xeon (Diamond Rapids): 81.07% without SmoothQuant, 82.54% with
`smooth_quant_alpha=0.5` (bf16: 82.17%), with the same oneDNN kernels and throughput.

## Examples

Examples of how to quantize a model and use a pre-quantized model can be found below.

- Usage examples shown on simple Keras model:
  - [Simple model](../../examples/jax/keras/simple_model/README.md)
- Examples of quantizing real Keras models using Neural Compressor
  - [Gemma3](../../examples/jax/keras/gemma/README.md)
  - [ViT](../../examples/jax/keras/vit/README.md)

## Backend and Device

Although Intel® Neural Compressor can run on any platform supporting 8-bit floating point with Keras using the JAX backend,
performance improvements from quantization will be visible on Intel® Xeon® processors
(with AMX-FP8 extension) with JAX version greater than [v0.9](https://github.com/jax-ml/jax/releases/tag/jax-v0.9.0)
(see the full [JAX releases](https://github.com/jax-ml/jax/releases) page).
