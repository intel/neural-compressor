Keras simple model quantization
============

This document describes quantization of a simple Keras model using Neural Compressor on Intel® Xeon® processors.

- [Create Environment](#1-create-environment)
- [Install modules](#2-install-modules)
- [Quantize model](#3-quantize-model)
- [Save and load quantized model](#4-save-and-load-quantized-model)
- [Quantization configs](#5-quantization-configs)
    - [White list and exclude list](#white-list-and-exclude-list)
    - [Composable configs](#composable-configs)
- [Configuration examples](#6-configuration-examples)
    - [Composable config](#composable-config)
    - [Load configuration from file](#load-configuration-from-file)
- [Some debug](#7-some-debug)

## 1. Create Environment
It is worth conducting experiments in a separate environment. For example, you can use the conda environment from [conda-forge](https://github.com/conda-forge/miniforge). The binary for your environment could be found here: [miniforge](https://github.com/conda-forge/miniforge/releases/latest)

## 2. Install modules
You can install Neural Compressor from pypi.org:
```bash
pip install neural_compressor_jax
```

Alternatively you can install it directly from source:
```bash
pushd ../../../..  # go to the root directory of the Neural Compressor source code
INC_JAX_ONLY=1 pip install .
# or with -e option for better debugging
#INC_JAX_ONLY=1 pip install -e .
popd
```

## 3. Quantize model

To quantize the model you have to make 3 steps:

1. Create the original model:
```python
class DummyModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense1 = keras.layers.Dense(10, activation="linear")
        self.dense2 = keras.layers.Dense(1, activation="linear")

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

model = DummyModel()
```

2. Calibrate the model using a dataset similar to the one that will be used later. In our example - we use random data. We can choose which floating point format will be used in the quantized model.

```python
from neural_compressor.jax import quantize_model, StaticQuantConfig

config = StaticQuantConfig(weight_dtype="fp8_e4m3", activation_dtype="fp8_e4m3")

def calib_function(model):
    key = jax.random.PRNGKey(1)
    input = 10 * jax.random.normal(key, (1, 32))
    model(input)

q_model = quantize_model(model, config, calib_function)
```

3. Use the quantized model
```python
quantized_output = q_model(input)
print(f"Quantized model output: {quantized_output}")
```

Working examples are included in this repository. You can run helloworld with this snippet:
```bash
python ../helloworld.py
```

Or different example for static/dynamic quantization comparison
```bash
python simple_config.py
```

## 4. Save and load quantized model

Calibration costs time, so we can calibrate once on representative data sets and later reuse it many times. To achieve it saving model functionality is supported.
You can run the [model_saving.py](model_saving.py) script:
```bash
python model_saving.py
```

The script quantizes the model, saves it to `./qmodel.keras`, then loads it back and verifies that the outputs of the freshly quantized model and the reloaded one match:

```python
keras.models.save_model(q_model, "./qmodel.keras")
loaded_model = keras.models.load_model("./qmodel.keras")

loaded_output = loaded_model(input)
match = jnp.allclose(quantized_output, loaded_output)
print(f"Results match: {match}")
```

Note that the model class is registered with `@register_keras_serializable` so that Keras can serialize/deserialize it by name when saving and loading.

## 5. Quantization configs

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
loaded_quant_config = JaxBaseConfig.from_json_file("path/to/quant_config.json")
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

It's possible to combine multiple configs to achieve per-layer and per-class configuration, by combining multiple configs into a *JaxComposableConfig*

***NOTE***
The order in which the configs are provided matters - multiple configs that apply for the same layer are resolved in a last-wins manner, as explained below.

```python
from neural_compressor.jax import StaticQuantConfig, DynamicQuantConfig

static1 = StaticQuantConfig(weight_dtype="int8", activation_dtype="int8", white_list=["dense1", "dense2"])
dynamic = DynamicQuantConfig(weight_dtype="fp8_e4m3", activation_dtype="fp8_e4m3", white_list=["dense2", "dense3"])
static2 = StaticQuantConfig(weight_dtype="int8", activation_dtype="int8", white_list=["Dense"], exclude_list=["dense1", "dense2"])

# Compose the configurations into a single configuration
#
# This composition results in quantization being applied for layers in order in which configs are constructed:
# * ``dense1``  -> only ``static1`` matches                -> static
# * ``dense2``  -> ``static1`` and ``dynamic`` match       -> dynamic (later wins)
# * ``dense3``  -> ``dynamic`` and ``static2`` match       -> static  (later wins)
config = static1 + dynamic + static2
```

## 6. Configuration examples

### Composable config

Neural Compressor allows composing several quantization configurations together, so that different parts of the model (matched via `white_list`/`exclude_list`) can use different quantization modes (static/dynamic) and dtypes. This is demonstrated in [composable_config.py](composable_config.py):

```python
config1 = StaticQuantConfig(
    weight_dtype="fp8_e4m3",
    activation_dtype="fp8_e4m3",
    white_list=["dense.*"],
    exclude_list=["dense3"],
)
config2 = DynamicQuantConfig(
    weight_dtype="fp8_e5m2",
    activation_dtype="fp8_e5m2",
    white_list=["dense3", "dense4"],
)

# Dynamic quantization will be applied to dense3 and dense4,
# while static quantization will be used for the remaining dense layers.
#
# Additionally dense4 is being matched by both static and dynamic config.
# Decision which one will be actually used depends on order in which configs are applied (later wins).
# In below example config2 is applied after config1 so dense4 will be dynamically quantized.
# note: You can see warning about config override with LOGLEVEL=DEBUG
composable_config = config1 + config2
```

Run it with:
```bash
python composable_config.py
```

### Load configuration from file

Instead of constructing the quantization configuration directly in Python, you can store it in a JSON file and load it at runtime. This is demonstrated in [external_config.py](external_config.py), which reads the configuration with:

```python
config = JaxBaseConfig.from_json_file(args.quant_config_file)
```

The example configurations are available in the local [configs](configs) directory:

- [configs/static_config.json](configs/static_config.json) for static quantization
- [configs/dynamic_config.json](configs/dynamic_config.json) for dynamic quantization
- [configs/composable_config.json](configs/composable_config.json) for a composed multi-rule setup

For example, a static quantization configuration looks like this:

```json
{
    "quantization_type": "static_quant",
    "config": {
        "weight_dtype": "fp8_e4m3",
        "activation_dtype": "fp8_e4m3",
        "const_scale": true,
        "const_weight": false,
        "weight_scale_granularity": "per_tensor",
        "dot_product_attention_enable": false
    }
}
```

Run the example by passing the JSON file path on the command line:

```bash
python external_config.py --quant_config_file configs/static_config.json
```

You can swap in `configs/dynamic_config.json` or `configs/composable_config.json` to try other quantization modes. Note that when the JSON describes only dynamic quantization, the calibration function defined in the script is not used.

## 7. Some debug

If you are interested how your model looks like after quantization, all the example scripts already print the flattened layer list before and after quantization:

```python
print("Quantized model layers:")
for layer in q_model._flatten_layers():
    print(layer)
```

For deeper insight (e.g. per-layer scales), you can additionally set the environment variable:
```bash
export LOGLEVEL=DEBUG
```
and use the `print_model()` utility on the quantized model, as demonstrated in the [Gemma](../gemma/README.html#6-some-debug) and [ViT](../vit/README.html#6-some-debug) examples.
