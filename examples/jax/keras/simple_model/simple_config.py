import os


# Set Keras backend to JAX before importing Keras
os.environ["KERAS_BACKEND"] = "jax"

import jax
import keras

from neural_compressor.jax import quantize_model, StaticQuantConfig, DynamicQuantConfig


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
    model = DummyModel()
    print("Original model layers:")
    _print_model_layers(model)

    # Prepare input data
    key = jax.random.PRNGKey(0)
    input = 5 * jax.random.normal(key, (1, 32))

    # Run the original model to get baseline output
    original_output = model(input)
    print(f"Original model output: {original_output}", "\n")

    # === Static quantization === #

    # Prepare QuantConfig with all default values being explicitly provided
    print("=== Static quantization ===")
    static_config = StaticQuantConfig(
        weight_dtype="fp8_e4m3",
        activation_dtype="fp8_e4m3",
        const_scale=True,
        const_weight=True,
        weight_scale_granularity="per_tensor",
        dot_product_attention_enable=False
    )

    # Define a calibration function
    # The calibration function runs the model with representative data to collect statistics
    # for static quantization.
    def calib_function(model):
        key = jax.random.PRNGKey(1)
        input = 10 * jax.random.normal(key, (1, 32))
        model(input)

    # Quantize the model
    print("Quantizing model", '\n')
    qs_model = quantize_model(model, static_config, calib_function, inplace=False)
    
    # Print quantized model layers
    print("Quantized model layers:")
    _print_model_layers(qs_model)

    # Run the quantized model
    quantized_output = qs_model(input)
    print(f"Quantized model output: {quantized_output}", "\n")

    # === Dynamic quantization === #

    print("=== Dynamic quantization ===")
    # Prepare QuantConfig with all default values being explicitly provided
    dynamic_config = DynamicQuantConfig(
        weight_dtype="fp8_e4m3",
        activation_dtype="fp8_e4m3",
        const_scale=True,
        const_weight=True,
        weight_scale_granularity="per_tensor",
        dot_product_attention_enable=False
    )
    # Calibration function is not needed for dynamic quantization

    # Quantize the model
    print("Quantizing model", '\n')
    qd_model = quantize_model(model, dynamic_config, inplace=False)
    
    # Print quantized model layers
    print("Quantized model layers:")
    _print_model_layers(qd_model)

    # Run the quantized model
    quantized_output = qd_model(input)
    print(f"Quantized model output: {quantized_output}", "\n")


if __name__ == "__main__":
    main()
