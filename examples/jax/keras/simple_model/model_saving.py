import os

# Set Keras backend to JAX before importing Keras
os.environ["KERAS_BACKEND"] = "jax"

import jax
import jax.numpy as jnp
import keras
from keras.saving import register_keras_serializable
from neural_compressor.jax import quantize_model, StaticQuantConfig


@register_keras_serializable(package="EXAMPLE", name=None)
class DummyModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense1 = keras.layers.Dense(10, activation="linear")
        self.dense2 = keras.layers.Dense(1, activation="linear")

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)


def _print_model_layers(model):
    for layer in model._flatten_layers():
        print(layer)
    print()


def main():
    # Set random seed for reproducibility - generate always the same weights
    keras.utils.set_random_seed(473)

    # Define a simple Keras model
    print("Creating model...")
    model = DummyModel()

    # Print model layers
    print("Original model layers:")
    _print_model_layers(model)

    # Prepare input data
    key = jax.random.PRNGKey(0)
    input = 5 * jax.random.normal(key, (1, 32))

    # Run the original model to get baseline output
    original_output = model(input)
    print(f"Original model output: {original_output}")

    # Prepare QuantConfig
    config = StaticQuantConfig(weight_dtype="fp8_e4m3", activation_dtype="fp8_e4m3")

    # Define a calibration function
    # The calibration function runs the model with representative data to collect statistics
    # for static quantization.
    def calib_function(model):
        key = jax.random.PRNGKey(1)
        input = 10 * jax.random.normal(key, (1, 32))
        model(input)

    # Quantize the model
    print("Quantizing model...")
    q_model = quantize_model(model, config, calib_function)

    # Print quantized model layers
    print("Quantized model layers:")
    _print_model_layers(q_model)

    # Run the quantized model
    quantized_output = q_model(input)
    print(f"Quantized model output: {quantized_output}")

    # Save and load the quantized model
    path_to_saved_model = "./qmodel.keras"

    print(f"Saving quantized model to {path_to_saved_model}...")
    keras.models.save_model(q_model, path_to_saved_model)

    print("Original quant config:")
    print(q_model._quant_config)

    print(f"Loading quantized model from {path_to_saved_model}...")
    loaded_model = keras.models.load_model(path_to_saved_model)

    print("Loaded quantized model layers:")
    _print_model_layers(loaded_model)

    print("Loaded quant config:")
    print(loaded_model._quant_config)

    # Run the loaded model
    loaded_output = loaded_model(input)
    print(f"Loaded model output: {loaded_output}")

    # Verify results match
    match = jnp.allclose(quantized_output, loaded_output)
    print(f"Results match: {match}")


if __name__ == "__main__":
    main()
