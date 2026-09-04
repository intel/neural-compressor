import os


# Set Keras backend to JAX before importing Keras
os.environ["KERAS_BACKEND"] = "jax"

import argparse

import jax
import jax.numpy as jnp
import keras

from neural_compressor.jax import quantize_model, JaxBaseConfig, StaticQuantConfig


class DummyModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense1 = keras.layers.Dense(10, activation="linear")
        self.dense2 = keras.layers.Dense(1, activation="linear")

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

parser = argparse.ArgumentParser("Quantize and use a simple Keras model")
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
print("Original model layers:")
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
    print("\nUsing default quantization configuration")
    config = StaticQuantConfig(weight_dtype="fp8_e4m3", activation_dtype="fp8_e4m3")

print(f"\nQuant config:\n{config}")

if config.name == "dynamic_quant":
    print("\nDynamic quantization does not require calibration. Skipping calibration step.")

# The calibration function runs the model with representative data to collect statistics
# for static quantization. This has no effect on dynamic quantization.
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
