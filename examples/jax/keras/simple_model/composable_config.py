import os


# Set Keras backend to JAX before importing Keras
os.environ["KERAS_BACKEND"] = "jax"

import jax
import keras

from neural_compressor.jax import quantize_model, StaticQuantConfig, DynamicQuantConfig


class DummyModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense1 = keras.layers.Dense(6, activation="linear", name="dense1")
        self.dense2 = keras.layers.Dense(4, activation="linear", name="dense2")
        self.dense3 = keras.layers.Dense(4, activation="linear", name="dense3")
        self.dense4 = keras.layers.Dense(2, activation="linear", name="dense4")

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.dense4(x)


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

    # Static quantization will be applied for all dense layers excluding dense3 and dense4
    config1 = StaticQuantConfig(
        weight_dtype="fp8_e4m3",
        activation_dtype="fp8_e4m3",
        const_scale=True,
        const_weight=True,
        weight_scale_granularity="per_tensor",
        dot_product_attention_enable=False,
        white_list=["dense.*"],
        exclude_list=["dense3"]
    )

    # Dynamic quantization will be applied only for dense3 and dense4
    config2 = DynamicQuantConfig(
            weight_dtype="fp8_e5m2",
            activation_dtype="fp8_e5m2",
            const_scale=True,
            const_weight=True,
            weight_scale_granularity="per_tensor",
            dot_product_attention_enable=False,
            white_list=["dense3", "dense4"]
        )

    # Merge both configs into a single composable config
    composable_config = config1 + config2

    # Define a calibration function
    # The calibration function runs the model with representative data to collect statistics
    # for static quantization.
    def calib_function(model):
        key = jax.random.PRNGKey(1)
        input = 10 * jax.random.normal(key, (1, 32))
        model(input)

    # Quantize the model. Different QuantConfigs will be applies for different layers
    print("Quantizing model", '\n')
    q_model = quantize_model(model, composable_config, calib_function, inplace=True)
    
    # Print quantized model layers
    print("Quantized model layers:")
    _print_model_layers(q_model)

    # Run the quantized model
    quantized_output = q_model(input)
    print(f"Quantized model output: {quantized_output}", "\n")


if __name__ == "__main__":
    main()
