import os

os.environ["KERAS_BACKEND"] = "jax"

from keras_hub.models import Gemma3CausalLM
import argparse
import neural_compressor.jax  # Required to load quantized model

parser = argparse.ArgumentParser("Run statically quantized Gemma3CausalLM model")
parser.add_argument(
    "-q",
    "--quantized_path",
    default="/tmp/gemma3_instruct_270m_quantized",
    type=str,
    help="path to quantized model",
)
args = parser.parse_args()

print("Load quantized model from:", args.quantized_path)
gemma_lm = Gemma3CausalLM.from_preset(args.quantized_path)
gemma_lm.summary()

print("Generate output using quantized model")
output = gemma_lm.generate({"prompts": "Describe the city of Berlin?"}, max_length=100)
print("Output:\n", output)
