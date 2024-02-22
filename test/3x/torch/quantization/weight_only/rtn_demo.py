# ! Rename it

import pytest
import torch
import transformers

from neural_compressor.torch.quantization import RTNConfig, get_default_rtn_config, quantize


class TestRTNQuant:
    def test_rtn(self):
        self.tiny_gptj = transformers.AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-GPTJForCausalLM",
        )
        self.example_inputs = torch.tensor([[10, 20, 30, 40, 50, 60]], dtype=torch.long)
        model = self.tiny_gptj
        # record label for comparison
        self.label = model(self.example_inputs.to(model.device))[0]
        # test_default_config

        quant_config = get_default_rtn_config()
        q_model = quantize(model, quant_config)
        # record q_label for comparison
        self.q_label = model(self.example_inputs.to(q_model.device))[0]


t = TestRTNQuant()
t.test_rtn()
