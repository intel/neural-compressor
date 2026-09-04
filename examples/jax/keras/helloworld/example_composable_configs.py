import os


# Set Keras backend to JAX before importing Keras
os.environ["KERAS_BACKEND"] = "jax"

import argparse

import jax
import jax.numpy as jnp
import keras

from neural_compressor.jax import (
    quantize_model,
    JaxBaseConfig,
    StaticQuantConfig,
    DynamicQuantConfig,
    JaxComposableConfig,
)


class DummyModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense1 = keras.layers.Dense(6, activation="linear", name="dense1")
        self.dense2 = keras.layers.Dense(4, activation="linear", name="dense2")
        self.dense3 = keras.layers.Dense(2, activation="linear", name="dense3")

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)


parser = argparse.ArgumentParser("Quantize and use a simple Keras model with composable configs")
parser.add_argument(
    "--quant_config_file",
    default=None,
    help="Path to the quantization configuration file.\nExample configs can be found under *../configs* directory.",
)

args = parser.parse_args()


print("Creating model...")
# Set random seed for reproducibility - generate always the same weights
keras.utils.set_random_seed(473)
model = DummyModel()

# Print model layers
print("\nOriginal model layers:")
for layer in model._flatten_layers():
    print(layer)
print()

key = jax.random.PRNGKey(0)
# Generate random input data
data = 5 * jax.random.normal(key, (1, 32))
print("\nRandom input data:\n", data)

# Run the original model to get baseline output
original_output = model(data)
print(f"\nOriginal model output: {original_output}")

if args.quant_config_file:
    print(f"\nLoading quantization configuration from: {args.quant_config_file}")
    config = JaxBaseConfig.from_json_file(args.quant_config_file)
else:
    print("\nUsing default composable quantization configuration")
    # Define quantization configurations.
    # We can mix static and dynamic quantization, different dtypes and other options.
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
    # IMPORTANT The order of the configs matters, as the last matching config will be used for each layer.
    config = static1 + dynamic + static2

print(f"\nQuant config:\n{config}")

# The calibration function runs the model with representative data to collect statistics
# for static quantization.
# When mixing static and dynamic quantization, the calibration data is collected only for the layers that are
# statically quantized - the remaining layers are kept in the original precision.
# So the order is: static quantization mechanism -> dynamic quantization mechanism.
def calib_function(model):
    # Run inference on a few batches of data
    model(jnp.zeros((1, 32)))
    model(15 * jnp.ones((1, 32)))

print("\nQuantizing model...")
q_model = quantize_model(model, config, calib_function)

# Print quantized model layers
print("\nQuantized model layers:")
for layer in q_model._flatten_layers():
    print(layer)
print()

quantized_output = q_model(data)

print(f"\nQuantized model output: {quantized_output}")
