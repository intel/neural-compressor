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


def main():
    # Set random seed for reproducibility
    keras.utils.set_random_seed(473)

    path_to_saved_model = "./qmodel.keras"

    # 1. Define a simple Keras model
    print("Creating model...")
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

    # 7. Save and load the quantized model
    print(f"Saving quantized model to {path_to_saved_model}...")
    keras.models.save_model(q_model, path_to_saved_model)

    print("Original quant config:")
    print(q_model._quant_config)

    print(f"Loading quantized model from {path_to_saved_model}...")
    loaded_model = keras.models.load_model(path_to_saved_model)

    print("Loaded quantized model layers:")
    for layer in loaded_model._flatten_layers():
        print(layer)
    print()

    print("Loaded quant config:")
    print(loaded_model._quant_config)

    # 8. Run the loaded model
    loaded_output = loaded_model(data)
    print(f"Loaded model output: {loaded_output}")

    # Verify results match
    match = jnp.allclose(quantized_output, loaded_output)
    print(f"Results match: {match}")


if __name__ == "__main__":
    main()
