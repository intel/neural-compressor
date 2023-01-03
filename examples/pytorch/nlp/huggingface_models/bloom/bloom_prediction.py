import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument("--prompt", default='Translate to English: Je t’aime.', type=str, help="prompt")

args = parser.parse_args()

from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "bloomz-7b1"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

inputs = tokenizer.encode(args.prompt, return_tensors="pt")

# FP32 prediction
outputs = model.generate(inputs)
print("Prediction of the original model: ", tokenizer.decode(outputs[0]))

from neural_compressor.conf.config import QuantConf
from neural_compressor.experimental import Quantization, common
quant_config = QuantConf()
quant_config.usr_cfg.quantization.approach = "post_training_dynamic_quant"
quant_config.usr_cfg.model.framework = "pytorch"
quantizer = Quantization(quant_config)
quantizer.model = common.Model(model)
model = quantizer()
model = model.model
model.eval()

# INT8 prediction (dynamic quantization)
outputs = model.generate(inputs)
print("Prediction of the quantized model: ", tokenizer.decode(outputs[0]))