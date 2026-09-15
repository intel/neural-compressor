import os


# Set Keras backend to JAX before importing Keras
os.environ["KERAS_BACKEND"] = "jax"

import argparse
import jax
import keras

from neural_compressor.jax import quantize_model, JaxBaseConfig


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
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quant_config_file",
        required=True,
        help="Path to the quantization configuration file.\nExample configs can be found under \"examples/jax/keras/simple_model/configs\" directory.",
    )
    args = parser.parse_args()

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

    # Read QuantConfig from json file
    config = JaxBaseConfig.from_json_file(args.quant_config_file)

    # Define a calibration function
    # JSON config may contain dynamic quantization for all layers in which case this function will not be used
    def calib_function(model):
        key = jax.random.PRNGKey(1)
        input = 10 * jax.random.normal(key, (1, 32))
        model(input)

    # Quantize the model
    print("Quantizing model", '\n')
    q_model = quantize_model(model, config, calib_function, inplace=True)
    
    # Print quantized model layers
    print("Quantized model layers:")
    _print_model_layers(q_model)

    # Run the quantized model
    quantized_output = q_model(input)
    print(f"Quantized model output: {quantized_output}", "\n")


if __name__ == "__main__":
    main()
