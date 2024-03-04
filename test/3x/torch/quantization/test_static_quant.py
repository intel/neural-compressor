import copy

import pytest
import torch

from neural_compressor.torch.quantization import StaticQuantConfig, get_default_static_config, quantize
from neural_compressor.torch.utils import is_ipex_available

if is_ipex_available():
    import intel_extension_for_pytorch as ipex


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


class TestStaticQuant:
    def setup_class(self):
        self.fp32_model = build_simple_torch_model()
        self.input = torch.randn(1, 30)

    def teardown_class(self):
        pass

    @pytest.mark.skipif(not is_ipex_available(), reason="Requires IPEX")
    def test_static_quant_default(self):
        fp32_model = copy.deepcopy(self.fp32_model)
        quant_config = get_default_static_config()
        example_inputs = self.input
        q_model = quantize(fp32_model, quant_config=quant_config, run_fn=run_fn, example_inputs=example_inputs)
        assert q_model is not None, "Quantization failed!"

    @pytest.mark.skipif(not is_ipex_available(), reason="Requires IPEX")
    @pytest.mark.parametrize(
        "act_sym, act_algo",
        [
            (True, "kl"),
            (True, "minmax"),
            (False, "kl"),
            (False, "minmax"),
        ],
    )
    def test_static_quant_params(self, act_sym, act_algo):
        fp32_model = copy.deepcopy(self.fp32_model)
        quant_config = StaticQuantConfig(act_sym=act_sym, act_algo=act_algo)
        example_inputs = self.input
        q_model = quantize(fp32_model, quant_config=quant_config, run_fn=run_fn, example_inputs=example_inputs)
        assert q_model is not None, "Quantization failed!"

    @pytest.mark.skipif(not is_ipex_available(), reason="Requires IPEX")
    def test_static_quant_accuracy(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2, False)

            def forward(self, x):
                x = self.linear(x)
                x = x + x
                return x

        model = M()

        def run_fn(model):
            model(torch.randn(3, 2))

        fp32_model = copy.deepcopy(model)
        fp32_model.linear.weight = torch.nn.Parameter(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
        example_inputs = torch.zeros(3, 2)
        quant_config = StaticQuantConfig(act_sym=True, act_algo="kl")
        q_model = quantize(fp32_model, quant_config=quant_config, run_fn=run_fn, example_inputs=example_inputs)
        output1 = fp32_model(example_inputs)
        output2 = q_model(example_inputs)
        # set a big atol to avoid random issue
        assert torch.allclose(output1, output2, atol=2e-2), "Accuracy gap atol > 0.02 is unexpected. Please check."
