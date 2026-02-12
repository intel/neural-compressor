import os


# Set Keras backend to JAX before importing Keras
os.environ["KERAS_BACKEND"] = "jax"

import jax
import jax.numpy as jnp
import keras

from neural_compressor.jax import quantize_model, StaticQuantConfig


class DummyModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense1 = keras.layers.Dense(10, activation="linear")
        self.dense2 = keras.layers.Dense(1, activation="linear")

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)


def main():
    # 1. Define a simple Keras model
    print("Creating model...")
    # Set random seed for reproducibility - generate always the same weights
    keras.utils.set_random_seed(473)
    model = DummyModel()

    # Print model layers
    print("Original model layers:")
    for layer in model._flatten_layers():
        print(layer)
    print()
    # 2. Prepare input data
    key = jax.random.PRNGKey(0)
    # Generate random input data
    data = 5 * jax.random.normal(key, (1, 32))
    print(data)

    # Run the original model to get baseline output
    original_output = model(data)
    print(f"Original model output: {original_output}")

    # 3. Define quantization configuration
    # We use FP8 (E4M3) for both weights and activations
    config = StaticQuantConfig(weight_dtype="fp8_e4m3", activation_dtype="fp8_e4m3")

    # 4. Define a calibration function
    # The calibration function runs the model with representative data to collect statistics
    # for static quantization.
    def calib_function(model):
        # Run inference on a few batches of data
        model(jnp.zeros((1, 32)))
        model(15 * jnp.ones((1, 32)))

    # 5. Quantize the model
    print("Quantizing model...")
    q_model = quantize_model(model, config, calib_function)
    # Print quantized model layers
    print("Quantized model layers:")
    for layer in q_model._flatten_layers():
        print(layer)
    print()
    # 6. Run the quantized model
    # JIT compile the quantized model for performance
    quantized_output = q_model(data)

    print(f"Quantized model output: {quantized_output}")


if __name__ == "__main__":
    main()
