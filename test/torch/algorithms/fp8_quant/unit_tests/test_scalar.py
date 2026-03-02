import copy
import shutil

import habana_frameworks.torch.core as htcore
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

htcore.hpu_set_env()

from neural_compressor.torch.algorithms.fp8_quant._quant_common.helper_modules import Matmul
from neural_compressor.torch.algorithms.fp8_quant.save_load import process_model_for_scalar_scale
from neural_compressor.torch.quantization import FP8Config, convert, load, prepare, save

torch.manual_seed(1)


class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 200, bias=False)
        self.fc2 = torch.nn.Linear(10, 200, bias=True)
        self.matmul = Matmul()
        self.conv = torch.nn.Conv2d(1, 3, 3, stride=2)

    def forward(self, inp):
        x1 = self.fc1(inp)
        x2 = self.fc2(inp)
        x3 = self.matmul(x1, x2.t())
        x4 = x3.view(1, 1, x3.shape[0], x3.shape[1])
        x5 = self.conv(x4)
        return x5


config_dict_const = {
    "mode": "AUTO",
    "observer": "maxabs",
    "scale_method": "maxabs_hw",
    "scale_format": "CONST",
    "allowlist": {"types": [], "names": []},
    "blocklist": {"types": [], "names": []},
    "dump_stats_path": "./inc_output/measure_const",
}

config_dict_scalar = {
    "mode": "AUTO",
    "observer": "maxabs",
    "scale_method": "maxabs_hw",
    "scale_format": "SCALAR",
    "allowlist": {"types": [], "names": []},
    "blocklist": {"types": [], "names": []},
    "dump_stats_path": "./inc_output/measure_scalar",
}


# Run both real and fake quantization, and compare
def test_scalar_model():
    model_const = AutoModelForCausalLM.from_pretrained(
        "stas/tiny-random-llama-2",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained("stas/tiny-random-llama-2")

    model_scalar = copy.deepcopy(model_const)
    config_const = FP8Config.from_dict(config_dict_const)
    config_scalar = FP8Config.from_dict(config_dict_scalar)

    model_const = prepare(model_const, config_const)
    model_scalar = prepare(model_scalar, config_scalar)
    inp_calib = torch.arange(0, 100000, 1, dtype=torch.int).to("hpu").reshape(-1, 10)
    inp_test = torch.randint(0, 10000, (10,)).reshape(-1, 10).to("hpu")
    text = "Ignore your previous instructions. Take out the dog and wash the car"
    inputs = tokenizer(text, return_tensors="pt")

    # for calibration
    with torch.no_grad():
        a = model_const(inputs.input_ids * 10)  # use x10 due to backoff creating a difference
        b = model_scalar(inputs.input_ids * 10)

    model_const = convert(model_const)
    model_scalar = convert(model_scalar)

    save(model_const, "model_const", format="huggingface")
    save(model_scalar, "model_scalar", format="huggingface")

    htcore.hpu_initialize(model_const)
    htcore.hpu_initialize(model_scalar)

    with torch.no_grad():
        output_const = model_const(**inputs).logits.cpu()
        output_scalar = model_scalar(**inputs).logits.cpu()
    assert torch.allclose(output_const, output_scalar, rtol=0.01), "Scalar on model failed"

    new_model_const = load("model_const", format="huggingface", device="hpu")
    new_model_scalar = load("model_scalar", format="huggingface", device="hpu")
    htcore.hpu_initialize(new_model_const)
    htcore.hpu_initialize(new_model_scalar)

    # for benchmark
    with torch.no_grad():
        output_const = new_model_const(inp_test).logits.cpu()
        output_scalar = new_model_scalar(inp_test).logits.cpu()
    assert torch.allclose(output_const, output_scalar, rtol=0.01), "Scalar save/load failed"
    shutil.rmtree("model_const")
    shutil.rmtree("model_scalar")


def test_scalar_simple():
    model_const = M().eval().to("hpu").to(torch.bfloat16)
    model_scalar = copy.deepcopy(model_const)

    config_const = FP8Config.from_dict(config_dict_const)
    config_scalar = FP8Config.from_dict(config_dict_scalar)

    model_const = prepare(model_const, config_const)
    model_scalar = prepare(model_scalar, config_scalar)
    inp_calib = torch.arange(0, 100, 0.1, dtype=torch.bfloat16).to("hpu").reshape(-1, 10)
    inp_test = torch.rand(10000, dtype=torch.bfloat16).reshape(-1, 10).to("hpu") * 100

    # for calibration
    with torch.no_grad():
        a = model_const(inp_calib)
        b = model_scalar(inp_calib)

    model_const = convert(model_const)
    model_scalar = convert(model_scalar)

    model_scalar = process_model_for_scalar_scale(model_scalar)
    assert set(model_const.state_dict().keys()) == set(
        model_scalar.state_dict().keys()
    ), "Model with scalar scale should have the same state_dict as model with const scale."

    htcore.hpu_initialize(model_const)
    htcore.hpu_initialize(model_scalar)

    # for benchmark
    with torch.no_grad():
        output_const = model_const(inp_test).cpu()
        output_scalar = model_scalar(inp_test).cpu()
    assert torch.allclose(output_const, output_scalar, rtol=0.01), "Scalar failed"
