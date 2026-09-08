Keras simple model quantization

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

No external model download is required for this example. A tiny `DummyModel` composed of `Dense` layers is defined directly in the example scripts.

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

You can simply run one of the prepared scripts:
```bash
python ../helloworld.py
# or
python simple_config.py
```

## 5. Save and load quantized model

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

## 6. Composable configurations

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
# while static quantization will be used for the remaining matching layers.
composable_config = config1 + config2
```

Run it with:
```bash
python composable_config.py
```

## 7. Configuration as json file

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

## 8. Some debug

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
