import contextlib
import copy
import os

import pytest
import torch
import transformers
from transformers import AutoModelForCausalLM

from neural_compressor.torch.algorithms.weight_only.hqq.auto_accelerator import auto_detect_accelerator
from neural_compressor.torch.quantization import HQQConfig, get_default_hqq_config, quantize


class TestHQQCUDA:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires a GPU")
    def test_hqq_cuda(self):
        model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        example_inputs = torch.tensor(
            [[10, 20, 30, 40, 50, 60]], dtype=torch.long, device=auto_detect_accelerator().current_device()
        )
        if auto_detect_accelerator().name == "cpu":
            from neural_compressor.torch.algorithms.weight_only.hqq.config import hqq_global_option

            hqq_global_option.use_half = False
        # test_default_config
        quant_config = get_default_hqq_config()
        model = quantize(model, quant_config)
        q_label = model(example_inputs)[0]
        print(q_label)
