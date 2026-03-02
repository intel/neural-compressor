# Smooth Quant

- [Introduction](#introduction)
- [Usage](#usage)
  - [Using a Fixed `alpha`](#using-a-fixed-alpha)
  - [Determining the `alpha` through auto-tuning](#determining-the-alpha-through-auto-tuning)
- [Examples](#examples)


## Introduction

Quantization is a common compression operation to reduce memory and accelerate inference by converting the floating point matrix to an integer matrix. For large language models (LLMs) with gigantic parameters, the systematic outliers make quantification of activations difficult.  [SmoothQuant](https://arxiv.org/abs/2211.10438), a training free post-training quantization (PTQ) solution, offline migrates this difficulty from activations to weights with a mathematically equivalent transformation.

Please refer to the document of [Smooth Quant](smooth_quant.md) for detailed fundamental knowledge.

## Usage
There are two ways to apply smooth quantization: 1) using a fixed `alpha` for the entire model or 2) determining the `alpha` through auto-tuning.

### Using a Fixed `alpha`
To set a fixed alpha for the entire model, users can follow this example:

```python
from neural_compressor.tensorflow import SmoothQuantConfig, StaticQuantConfig

quant_config = [SmoothQuantConfig(alpha=0.5), StaticQuantConfig()]
q_model = quantize_model(output_graph_def, [sq_config, static_config], calib_dataloader)
```
The `SmoothQuantConfig` should be combined with `StaticQuantConfig` in a list because we still need to insert QDQ and apply pattern fusion after the smoothing process.


### Determining the `alpha` through auto-tuning
Users can search for the best `alpha`  for the entire model.The tuning process looks for the optimal `alpha` value from a list of `alpha` values provided by the user.

Here is an example:

```python
from neural_compressor.tensorflow import StaticQuantConfig, SmoothQuantConfig

custom_tune_config = TuningConfig(config_set=[SmoothQuantConfig(alpha=[0.5, 0.6, 0.7]), StaticQuantConfig()])
best_model = autotune(
    model="fp32_model",
    tune_config=custom_tune_config,
    eval_fn=eval_fn_wrapper,
    calib_dataloader=calib_dataloader,
)
```
> Please note that, it may a considerable amount of time as the tuning process applies each `alpha` to the entire model and uses the evaluation result on the entire dataset as the metric to determine the best `alpha`.

## Examples

Users can also refer to [examples](https://github.com/intel/neural-compressor/blob/master/examples/tensorflow/nlp/large_language_models/quantization/ptq/smoothquant) on how to apply smooth quant to a TensorFlow model with `neural_compressor.tensorflow`.
