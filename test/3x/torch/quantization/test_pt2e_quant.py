import os
import unittest
from unittest.mock import patch

import pytest
import torch

from neural_compressor.common.utils import logger
from neural_compressor.torch.export import export
from neural_compressor.torch.quantization import (
    DynamicQuantConfig,
    StaticQuantConfig,
    convert,
    get_default_dynamic_config,
    get_default_static_config,
    prepare,
    quantize,
)
from neural_compressor.torch.utils import TORCH_VERSION_2_2_2, get_torch_version


@pytest.fixture
def force_not_import_ipex(monkeypatch):
    def _is_ipex_imported():
        return False

    monkeypatch.setattr("neural_compressor.torch.quantization.config.is_ipex_imported", _is_ipex_imported)
    monkeypatch.setattr("neural_compressor.torch.quantization.algorithm_entry.is_ipex_imported", _is_ipex_imported)
    monkeypatch.setattr("neural_compressor.torch.export._export.is_ipex_imported", _is_ipex_imported)


class TestPT2EQuantization:

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
        exported_model = export(model, example_inputs=example_inputs)
        return exported_model, example_inputs

    @pytest.mark.skipif(get_torch_version() <= TORCH_VERSION_2_2_2, reason="Requires torch>=2.3.0")
    def test_quantize_simple_model(self, force_not_import_ipex):
        model, example_inputs = self.build_simple_torch_model_and_example_inputs()
        quant_config = None

        def calib_fn(model):
            for i in range(2):
                model(*example_inputs)

        quant_config = get_default_static_config()
        q_model = quantize(model=model, quant_config=quant_config, run_fn=calib_fn)
        from torch._inductor import config

        config.freezing = True
        opt_model = torch.compile(q_model)
        out = opt_model(*example_inputs)
        logger.warning("out shape is %s", out.shape)
        assert out is not None

    @pytest.mark.skipif(get_torch_version() <= TORCH_VERSION_2_2_2, reason="Requires torch>=2.3.0")
    @pytest.mark.parametrize("is_dynamic", [False, True])
    def test_prepare_and_convert_on_simple_model(self, is_dynamic, force_not_import_ipex):
        model, example_inputs = self.build_simple_torch_model_and_example_inputs()
        quant_config = None

        def calib_fn(model):
            for i in range(2):
                model(*example_inputs)

        if is_dynamic:
            quant_config = get_default_dynamic_config()
        else:
            quant_config = get_default_static_config()

        prepared_model = prepare(model, quant_config=quant_config)
        calib_fn(prepared_model)
        q_model = convert(prepared_model)
        assert q_model is not None, "Quantization failed!"

        from torch._inductor import config

        config.freezing = True
        opt_model = torch.compile(q_model)
        out = opt_model(*example_inputs)
        logger.warning("out shape is %s", out.shape)
        assert out is not None

    @pytest.mark.skipif(get_torch_version() <= TORCH_VERSION_2_2_2, reason="Requires torch>=2.3.0")
    def test_prepare_and_convert_on_llm(self, force_not_import_ipex):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # set TOKENIZERS_PARALLELISM to false

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        model_name = "facebook/opt-125m"
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        input_ids = tokenizer("Hello, my dog is cute", return_tensors="pt")["input_ids"]
        example_inputs = (input_ids,)
        model = export(model, example_inputs=example_inputs)

        quant_config = get_default_static_config()
        # prepare
        prepare_model = prepare(model, quant_config)
        # calibrate
        for i in range(2):
            prepare_model(*example_inputs)
        # convert
        converted_model = convert(prepare_model)
        # inference
        from torch._inductor import config

        config.freezing = True
        opt_model = torch.compile(converted_model)
        out = opt_model(*example_inputs)
        assert out.logits is not None
