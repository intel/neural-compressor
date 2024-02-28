import copy

import pytest
import torch

from neural_compressor.torch.quantization import SmoothQuantConfig, get_default_sq_config, quantize
from neural_compressor.torch.utils import is_ipex_available

if is_ipex_available():
    import intel_extension_for_pytorch as ipex


class Model(torch.nn.Module):
    device = torch.device("cpu")

    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(3, 4)
        self.fc2 = torch.nn.Linear(4, 3)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out


model = Model()


def run_fn(model):
    model(torch.randn([1, 3]))


class TestSmoothQuant:
    @pytest.mark.skipif(not is_ipex_available(), reason="Requires IPEX")
    def test_smooth_quant_default(self):
        fp32_model = copy.deepcopy(model)
        quant_config = get_default_sq_config()
        example_inputs = torch.randn([1, 3])
        q_model = quantize(fp32_model, quant_config=quant_config, run_fn=run_fn, example_inputs=example_inputs)
        assert q_model is not None, "Quantization failed!"

    @pytest.mark.skipif(not is_ipex_available(), reason="Requires IPEX")
    @pytest.mark.parametrize(
        "act_sym, act_algo, alpha, folding, scale_sharing",
        [
            (True, "kl", 0.1, True, True),
            (True, "minmax", 0.1, False, False),
            (False, "kl", 0.5, True, False),
            (False, "minmax", 0.5, False, True),
            (True, "minmax", 0.1, False, True),
            (False, "kl", 0.5, True, False),
        ],
    )
    def test_sq_linear_params(self, act_sym, act_algo, alpha, folding, scale_sharing):
        fp32_model = copy.deepcopy(model)
        quant_config = SmoothQuantConfig(
            act_sym=act_sym, act_algo=act_algo, alpha=alpha, folding=folding, scale_sharing=scale_sharing
        )
        example_inputs = torch.randn([1, 3])
        q_model = quantize(fp32_model, quant_config=quant_config, run_fn=run_fn, example_inputs=example_inputs)
        assert q_model is not None, "Quantization failed!"

    @pytest.mark.skipif(not is_ipex_available(), reason="Requires IPEX")
    def test_smooth_quant_auto(self):
        fp32_model = copy.deepcopy(model)
        auto_alpha_args = {
            "alpha_min": 0.45,
            "alpha_max": 0.55,
            "alpha_step": 0.01,
            "shared_criterion": "mean",
            "do_blockwise": True,
        }
        quant_config = SmoothQuantConfig(alpha="auto", auto_alpha_args=auto_alpha_args, folding=False)
        example_inputs = torch.randn([1, 3])
        q_model = quantize(fp32_model, quant_config=quant_config, run_fn=run_fn, example_inputs=example_inputs)
        assert q_model is not None, "Quantization failed!"

    @pytest.mark.skipif(not is_ipex_available(), reason="Requires IPEX")
    def test_sq_accuracy(self):
        fp32_model = copy.deepcopy(model)
        quant_config = SmoothQuantConfig(act_sym=True, act_algo="kl", alpha="auto", folding=True, scale_sharing=True)
        example_inputs = torch.zeros([1, 3])
        q_model = quantize(fp32_model, quant_config=quant_config, run_fn=run_fn, example_inputs=example_inputs)
        assert q_model is not None, "Quantization failed!"
        output1 = model(example_inputs)
        output2 = q_model(example_inputs)
        assert torch.allclose(output1, output2, atol=2e-2), "Accuracy gap atol > 0.02 is unexpected. Please check."