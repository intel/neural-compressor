import copy

import pytest
import torch
import transformers

from neural_compressor.torch.algorithms.weight_only.hqq.auto_accelerator import auto_detect_accelerator
from neural_compressor.torch.quantization import HQQConfig, get_default_hqq_config, quantize


class TestHQQ:
    def test_hqq(self):
        self.tiny_gptj = transformers.AutoModelForCausalLM.from_pretrained(
            "/models/opt-125m",  # TODO: replace it with model name
        )
        example_inputs = torch.tensor(
            [[10, 20, 30, 40, 50, 60]], dtype=torch.long, device=auto_detect_accelerator().current_device()
        )
        # test_default_config
        model = copy.deepcopy(self.tiny_gptj)
        quant_config = get_default_hqq_config()
        model = quantize(model, quant_config)
        q_label = model(example_inputs)[0]
        print(q_label)
