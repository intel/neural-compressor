import torch
import transformers


model = transformers.AutoModelForCausalLM.from_pretrained(
    'hf-internal-testing/tiny-random-GPTJForCausalLM',
    torchscript=True,
)
lm_input = torch.ones([1, 10], dtype=torch.long)

def calib_func(model):
    model(lm_input)

from neural_compressor.adaptor.torch_utils.awq import _get_hidden_states

hid = _get_hidden_states(model, calib_func=calib_func)

print(hid)

