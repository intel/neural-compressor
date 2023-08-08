import torch
import transformers


model = transformers.AutoModelForCausalLM.from_pretrained(
    'hf-internal-testing/tiny-random-GPTJForCausalLM',
    torchscript=True,
)
lm_input = torch.ones([1, 10], dtype=torch.long)


class SimpleDataLoader():
    def __init__(self):
        self.batch_size = 1
        self.input = torch.randn([1, 32])

    def __iter__(self):
        for i in range(10):
            yield torch.ones([1, 10], dtype=torch.long)


def calib_func(model):
    model(lm_input)

from neural_compressor.adaptor.torch_utils.awq import _get_hidden_states

hid = _get_hidden_states(model, calib_func=calib_func)
hid = _get_hidden_states(model, dataloader=SimpleDataLoader())

print(hid)

