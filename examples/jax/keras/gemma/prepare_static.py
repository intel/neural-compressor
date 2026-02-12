import os

os.environ["KERAS_BACKEND"] = "jax"

from keras_hub.models import Gemma3CausalLM
from neural_compressor.jax import quantize_model, StaticQuantConfig
import argparse


parser = argparse.ArgumentParser("Quantize and run Gemma3CausalLM model")
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
    help="path to the Keras model (could be directory or url or registered keras model name)",
)
parser.add_argument(
    "-q",
    "--quantized_path",
    default="/tmp/gemma3_instruct_270m_quantized",
    type=str,
    help="path where to store quantized model.",
)
args = parser.parse_args()
print("Arguments:", *vars(args).items(), sep="\n")

print("\nLoad original model from:", args.model_path)
gemma_lm = Gemma3CausalLM.from_preset(args.model_path)
gemma_lm.summary()


print("Prepare quantization config")
config = StaticQuantConfig(weight_dtype=args.precision, activation_dtype=args.precision)


def calib_function(model):
    model.generate({"prompts": "Describe the city of Moscow"}, max_length=100)
    model.generate({"prompts": "What is a monkey?"}, max_length=100)


print("Start quantization")
gemma_lm = quantize_model(gemma_lm, config, calib_function)
gemma_lm.summary()

print("Save quantized model to:", args.quantized_path)
gemma_lm.save_to_preset(args.quantized_path)
