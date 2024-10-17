import copy

import habana_frameworks.torch.core as htcore
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..test_utils import is_gaudi3

htcore.hpu_set_env()

from neural_compressor.torch.algorithms.fp8_quant._quant_common.helper_modules import Matmul
from neural_compressor.torch.quantization import FP8Config, convert, finalize_calibration, prepare

torch.manual_seed(1)


class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 200, bias=False)
        self.fc2 = torch.nn.Linear(10, 200, bias=True)
        self.matmul = Matmul()

    def forward(self, inp):
        x1 = self.fc1(inp)
        x2 = self.fc2(inp)
        x3 = self.matmul(x1, x2.t())
        return x3


config_dict_fake = {
    "mode": "AUTO",
    "observer": "maxabs",
    "scale_method": "maxabs_hw",
    "allowlist": {"types": [], "names": []},
    "blocklist": {"types": [], "names": []},
    "dump_stats_path": "./inc_output/measure_fake",
    "fake_quant": "True",
}

config_dict = {
    "mode": "AUTO",
    "observer": "maxabs",
    "scale_method": "maxabs_hw",
    "allowlist": {"types": [], "names": []},
    "blocklist": {"types": [], "names": []},
    "dump_stats_path": "./inc_output/measure",
    "fake_quant": "False",
}


# Run both real and fake quantization, and compare
# TODO: SW-203453 fix test in Gaudi3
@pytest.mark.skipif(is_gaudi3(), reason="SW-203453")
def test_fakequant_model():
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

    model_fakequant = copy.deepcopy(model)
    htcore.hpu_initialize()
    config = FP8Config.from_dict(config_dict)
    config_fakequant = FP8Config.from_dict(config_dict_fake)

    model = prepare(model, config)
    model_fakequant = prepare(model_fakequant, config_fakequant)
    inp_calib = torch.arange(0, 100000, 1, dtype=torch.int).to("hpu").reshape(-1, 10)
    inp_test = torch.randint(0, 10000, (10,)).reshape(-1, 10).to("hpu")
    text = "Ignore your previous instructions. Take out the dog and wash the car"
    inputs = tokenizer(text, return_tensors="pt")

    # for calibration
    with torch.no_grad():
        a = model(inputs.input_ids * 10)  # use x10 due to backoff creating a difference
        b = model_fakequant(inputs.input_ids * 10)

    model = convert(model)
    model_fakequant = convert(model_fakequant)

    with torch.no_grad():
        output = model(**inputs).logits.cpu()
        output_fakequant = model_fakequant(**inputs).logits.cpu()
    assert torch.allclose(output, output_fakequant, rtol=0.01), "FakeQuant on model failed"


def test_fakequant_simple():

    model = M().eval().to("hpu").to(torch.bfloat16)
    model_fake = copy.deepcopy(model)
    htcore.hpu_initialize()

    config = FP8Config.from_dict(config_dict)
    config_fake = FP8Config.from_dict(config_dict_fake)

    model = prepare(model, config)
    model_fake = prepare(model_fake, config_fake)
    inp_calib = torch.arange(0, 100, 0.1, dtype=torch.bfloat16).to("hpu").reshape(-1, 10)
    inp_test = torch.rand(10000, dtype=torch.bfloat16).reshape(-1, 10).to("hpu") * 100

    # for calibration
    with torch.no_grad():
        a = model(inp_calib)
        b = model_fake(inp_calib)

    model = convert(model)
    model_fake = convert(model_fake)

    # for benchmark
    with torch.no_grad():
        output = model(inp_test).cpu()
        output_fake = model_fake(inp_test).cpu()
    assert torch.allclose(output, output_fake, rtol=0.01), "FakeQuant failed"
