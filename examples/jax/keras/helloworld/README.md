Keras Hello World model quantization

============

This document describes quantization of a simple Keras model using Neural Compressor on Intel® Xeon® processors. It is meant as a minimal, self-contained introduction to the INC JAX/Keras quantization API before moving on to real models such as [Gemma](../gemma/README.md) or [ViT](../vit/README.md).


## 1. Create Environment
It is worth conducting experiments in a separate environment. For example, you can use the conda environment from [conda-forge](https://github.com/conda-forge/miniforge). The binary for your environment could be found here: [miniforge](https://github.com/conda-forge/miniforge/releases/latest)

## 2. Install modules

Install Neural Compressor from the source code:
```bash
pushd ../../../..  # go to the root directory of the Neural Compressor source code
INC_JAX_ONLY=1 pip install .
popd
```

## 3. Model

No external model download is required for this example. A tiny `DummyModel` composed of two `Dense` layers is defined directly in the example scripts:

```python
class DummyModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense1 = keras.layers.Dense(10, activation="linear")
        self.dense2 = keras.layers.Dense(1, activation="linear")

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)
```

## 4. Quantize model

To quantize the model you have to make 3 steps:

1. Create the original model:
```python
model = DummyModel()
```

2. Calibrate the model using a dataset similar to the one that will be used later. In our example - we use random data. We can choose which floating point format will be used in the quantized model.

```python
from neural_compressor.jax import quantize_model, StaticQuantConfig

config = StaticQuantConfig(weight_dtype="fp8_e4m3", activation_dtype="fp8_e4m3")


def calib_function(model):
    model(jnp.zeros((1, 32)))
    model(15 * jnp.ones((1, 32)))


q_model = quantize_model(model, config, calib_function)
```

3. Use the quantized model
```python
quantized_output = q_model(data)
print(f"Quantized model output: {quantized_output}")
```

You can simply run this by running the prepared [example.py](example.py) or [example_static.py](example_static.py) scripts:
```bash
python example.py
```
`example.py` accepts an optional `--quant_config_file` argument pointing to a JSON quantization configuration file, such as those found under the [../configs](../configs) directory:
```bash
python example.py --quant_config_file ../configs/example_static_config.json
```

## 5. Save and load quantized model

Calibration costs time, so we can calibrate once on representative data sets and later reuse it many times. To achieve it saving model functionality is supported.
You can run the [example_saving.py](example_saving.py) script:
```bash
python example_saving.py
```

The script quantizes the model, saves it to `./qmodel.keras`, then loads it back and verifies that the outputs of the freshly quantized model and the reloaded one match:

```python
keras.models.save_model(q_model, "./qmodel.keras")
loaded_model = keras.models.load_model("./qmodel.keras")

loaded_output = loaded_model(data)
match = jnp.allclose(quantized_output, loaded_output)
print(f"Results match: {match}")
```

Note that the model class is registered with `@register_keras_serializable` so that Keras can serialize/deserialize it by name when saving and loading.

## 6. Composable configurations

Neural Compressor allows composing several quantization configurations together, so that different parts of the model (matched via `white_list`/`exclude_list`) can use different quantization modes (static/dynamic) and dtypes. This is demonstrated in [example_composable_configs.py](example_composable_configs.py):

```python
static1 = StaticQuantConfig(weight_dtype="int8", activation_dtype="int8", white_list=["dense1", "dense2"])
dynamic = DynamicQuantConfig(weight_dtype="fp8_e4m3", activation_dtype="fp8_e4m3", white_list=["dense2", "dense3"])
static2 = StaticQuantConfig(
    weight_dtype="int8", activation_dtype="int8", white_list=["Dense"], exclude_list=["dense1", "dense2"]
)

# The order of the configs matters, as the last matching config will be used for each layer.
config = static1 + dynamic + static2
```

Run it with:
```bash
python example_composable_configs.py
```
or with a composed configuration loaded from a JSON file, such as [../configs](../configs):
```bash
python example_composable_configs.py --quant_config_file ../configs/example_composable_config.json
```

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
and use the `print_model()` utility on the quantized model, as demonstrated in the [Gemma](../gemma/README.md#6-some-debug) and [ViT](../vit/README.md#6-some-debug) examples.
