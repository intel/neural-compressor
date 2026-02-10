import os

os.environ["KERAS_BACKEND"] = "jax"
# os.environ["LOGLEVEL"] = "DEBUG" # Uncomment to enable print_model

from keras_hub.models import Gemma3CausalLM
from neural_compressor.jax import quantize_model, StaticQuantConfig
from neural_compressor.jax.utils.utility import print_model
import argparse


parser = argparse.ArgumentParser("Quantize and use Gemma3CausalLM model")
parser.add_argument(
    "-p",
    "--precision",
    default="fp8_e4m3",
    type=str,
    choices=["fp8_e4m3", "fp8_e5m2"],
    help="precision for the model",
)
parser.add_argument(
    "-m",
    "--model_path",
    default="gemma3_instruct_270m",
    type=str,
    help="path to the Keras model (It could be directory or url or registered keras model name)",
)
args = parser.parse_args()


print("\nLoad original model from:", args.model_path)
gemma_lm = Gemma3CausalLM.from_preset(args.model_path)
print_model(gemma_lm)

output = gemma_lm.generate({"prompts": "Describe the city of Berlin?"}, max_length=100)
print("\nOutput before quantization:\n", output)

print("\nPrepare quantization config")
config = StaticQuantConfig(weight_dtype=args.precision, activation_dtype=args.precision)


def calib_function(model):
    model.generate({"prompts": "Describe the city of Moscow"}, max_length=100)


print("\nStart quantization")
gemma_lm = quantize_model(gemma_lm, config, calib_function)
print_model(gemma_lm)

output = gemma_lm.generate({"prompts": "Describe the city of Berlin?"}, max_length=100)
print("\nOutput after quantization:\n", output)
