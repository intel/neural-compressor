import copy

import intel_extension_for_pytorch as ipex
import pytest
import torch

from neural_compressor.torch.quantization import SmoothQuantConfig, get_default_sq_config, quantize
from neural_compressor.torch.utils import is_ipex_available


def build_simple_torch_model():
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.fc1 = torch.nn.Linear(30, 50)
            self.fc2 = torch.nn.Linear(50, 30)
            self.fc3 = torch.nn.Linear(30, 5)

        def forward(self, x):
            out = self.fc1(x)
            out = self.fc2(out)
            out = self.fc3(out)
            return out

    model = Model()
    return model


def run_fn(model):
    model(torch.rand((1, 30)))
    model(torch.rand((1, 30)))


class TestSmoothQuant:
    def setup_class(self):
        self.fp32_model = build_simple_torch_model()
        self.input = torch.randn(1, 30)

    def teardown_class(self):
        pass

    @pytest.mark.skipif(not is_ipex_available(), reason="Requires IPEX")
    def test_smooth_quant_default(self):
        fp32_model = copy.deepcopy(self.fp32_model)
        quant_config = get_default_sq_config()
        example_inputs = self.input
        q_model = quantize(fp32_model, quant_config=quant_config, run_fn=run_fn, example_inputs=example_inputs)
        assert q_model is not None, "Quantization failed!"

    @pytest.mark.skipif(not is_ipex_available(), reason="Requires IPEX")
    @pytest.mark.parametrize(
        "act_sym, act_algo, alpha, folding",
        [
            (True, "kl", 0.1, True),
            (True, "minmax", 0.1, False),
            (False, "kl", 0.5, True),
            (False, "minmax", 0.5, False),
        ],
    )
    def test_static_quant_params(self, act_sym, act_algo, alpha, folding):
        fp32_model = copy.deepcopy(self.fp32_model)
        quant_config = SmoothQuantConfig(act_sym=act_sym, act_algo=act_algo, alpha=alpha, folding=folding)
        example_inputs = self.input
        q_model = quantize(fp32_model, quant_config=quant_config, run_fn=run_fn, example_inputs=example_inputs)
        assert q_model is not None, "Quantization failed!"