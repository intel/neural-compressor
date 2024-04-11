import copy

import pytest
import torch

from neural_compressor.torch.quantization import MXQuantConfig, get_default_mx_config, quantize


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


class TestMXQuant:
    def setup_class(self):
        self.fp32_model = build_simple_torch_model()
        self.input = torch.randn(1, 30)

    def teardown_class(self):
        pass

    def test_mx_quant_default(self):
        fp32_model = copy.deepcopy(self.fp32_model)
        quant_config = get_default_mx_config()
        q_model = quantize(fp32_model, quant_config=quant_config)
        assert q_model is not None, "Quantization failed!"

    @pytest.mark.parametrize(
        "w_dtype, weight_only",
        [
            ("fp4", True),
            ("fp8_e5m2", False),
        ],
    )
    def test_mx_quant_params(self, w_dtype, weight_only):
        fp32_model = copy.deepcopy(self.fp32_model)
        quant_config = MXQuantConfig(w_dtype=w_dtype, weight_only=weight_only)
        q_model = quantize(fp32_model, quant_config=quant_config)
        assert q_model is not None, "Quantization failed!"

    def test_mx_quant_accuracy(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2, False)

            def forward(self, x):
                x = self.linear(x)
                x = x + x
                return x

        model = M()

        fp32_model = copy.deepcopy(model)
        fp32_model.linear.weight = torch.nn.Parameter(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
        example_inputs = torch.zeros(3, 2)
        quant_config = MXQuantConfig()
        q_model = quantize(fp32_model, quant_config=quant_config)
        output1 = fp32_model(example_inputs)
        output2 = q_model(example_inputs)
        # set a big atol to avoid random issue
        assert torch.allclose(output1, output2, atol=2e-2), "Accuracy gap atol > 0.02 is unexpected. Please check."
