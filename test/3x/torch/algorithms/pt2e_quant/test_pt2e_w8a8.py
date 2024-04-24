import unittest
from unittest.mock import patch

import pytest
import torch

from neural_compressor.common.utils import logger
from neural_compressor.torch.algorithms.pt2e_quant.core import W8A8StaticQuantizer
from neural_compressor.torch.utils import TORCH_VERSION_2_2_2, get_torch_version


class TestW8A8StaticQuantizer:

    @staticmethod
    def get_toy_model():
        class Bar(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                x = a / (torch.abs(a) + 1)
                if b.sum() < 0:
                    b = b * -1
                return x * b

        inp1 = torch.randn(10)
        inp2 = torch.randn(10)
        example_inputs = (inp1, inp2)
        bar = Bar()
        return bar, example_inputs

    @staticmethod
    def build_simple_torch_model_and_example_inputs():
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(10, 20)
                self.fc2 = torch.nn.Linear(20, 10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.fc1(x)
                x = torch.nn.functional.relu(x)
                x = self.fc2(x)
                return x

        model = SimpleModel()
        example_inputs = (torch.randn(10, 10),)
        return model, example_inputs

    @pytest.mark.skipif(get_torch_version() <= TORCH_VERSION_2_2_2, reason="Requires torch>=2.3.0")
    def test_quantizer_on_simple_model(self):
        model, example_inputs = self.build_simple_torch_model_and_example_inputs()
        quant_config = None
        w8a8_static_quantizer = W8A8StaticQuantizer()
        # prepare
        prepare_model = w8a8_static_quantizer.prepare(model, quant_config, example_inputs=example_inputs)
        # calibrate
        for i in range(2):
            prepare_model(*example_inputs)
        # convert
        converted_model = w8a8_static_quantizer.convert(prepare_model)
        # inference
        from torch._inductor import config

        config.freezing = True
        opt_model = torch.compile(converted_model)
        out = opt_model(*example_inputs)
        logger.warning("out shape is %s", out.shape)
        assert out is not None

    @pytest.mark.skipif(get_torch_version() <= TORCH_VERSION_2_2_2, reason="Requires torch>=2.3.0")
    def test_quantizer_on_llm(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "facebook/opt-125m"
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        input_ids = tokenizer("Hello, my dog is cute", return_tensors="pt")["input_ids"]
        example_inputs = (input_ids,)
        quant_config = None
        w8a8_static_quantizer = W8A8StaticQuantizer()
        # prepare
        prepare_model = w8a8_static_quantizer.prepare(model, quant_config, example_inputs=example_inputs)
        # calibrate
        for i in range(2):
            prepare_model(*example_inputs)
        # convert
        converted_model = w8a8_static_quantizer.convert(prepare_model)
        # inference
        from torch._inductor import config

        config.freezing = True
        opt_model = torch.compile(converted_model)
        out = opt_model(*example_inputs)
        assert out.logits is not None

    @patch("neural_compressor.torch.algorithms.pt2e_quant.core.logger.error")
    def test_export_model_failed(self, mock_error):
        model, example_inputs = self.get_toy_model()
        w8a8_static_quantizer = W8A8StaticQuantizer()
        # export model
        exported_model = w8a8_static_quantizer.export_model(model, example_inputs=example_inputs)
        assert exported_model is None
        call_args_list = mock_error.call_args_list
        print([info[0][0] for info in call_args_list])
        assert any(["Failed to export the model" in msg for msg in [info[0][0] for info in call_args_list]])
