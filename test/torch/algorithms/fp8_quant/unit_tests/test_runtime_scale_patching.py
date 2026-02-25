import copy
import os
import shutil

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.utils.experimental as htexp
import pytest
import torch

from neural_compressor.torch.algorithms.fp8_quant._core.common import is_runtime_scale_patching
from neural_compressor.torch.algorithms.fp8_quant._quant_common.quant_config import ScaleMethodString
from neural_compressor.torch.quantization import FP8Config, convert, finalize_calibration, prepare

from ..tester import RUNTIME_SCALE_PATCHING_SUPPORTED_METHODS_LIST, run_with_raised_exception

os.environ["PT_HPU_WEIGHT_SHARING"] = "0"
htcore.hpu_inference_set_env()


class TinyBlock(torch.nn.Module):

    def __init__(self):
        super(TinyBlock, self).__init__()
        self.pre_linear = torch.nn.Linear(2, 1, bias=False)
        self.pre_linear.weight = torch.nn.Parameter(torch.ones([1, 2]))

    def forward(self, x):
        x = self.pre_linear(x)
        return x


class TinyModel(torch.nn.Module):

    def __init__(self):
        super(TinyModel, self).__init__()
        self.block = TinyBlock()

    def forward(self, x):
        x = self.block(x)
        return x


@pytest.fixture
def temp_directory():
    # Create a temporary directory
    temp_dir = "./test_runtime_scale_patching_outputs"
    os.makedirs(temp_dir)
    # Yield the temporary directory path to the test
    yield temp_dir
    # Cleanup: Remove the temporary directory after the test ends
    shutil.rmtree(temp_dir)


@pytest.mark.skipif(not hasattr(htexp, "_set_scale_attributes"), reason="scale attributes not supported")
@pytest.mark.parametrize("scale_method", ScaleMethodString)
@pytest.mark.parametrize("scale_format", ["SCALAR", "CONST"])
@pytest.mark.parametrize("dynamic_scale_patching", [True, False])
def test_no_assert(scale_method, scale_format, dynamic_scale_patching, temp_directory):
    model = TinyModel()
    model.eval()
    model = model.to("hpu").to(torch.bfloat16)
    inference_model = copy.deepcopy(model)
    htcore.hpu_inference_initialize()

    measure_config_dict = {
        "mode": "MEASURE",
        "observer": "maxabs",
        "allowlist": {"types": [], "names": []},
        "blocklist": {"types": [], "names": []},
        "dump_stats_path": f"{temp_directory}/inc_output",
    }
    quant_config_dict = {
        "mode": "QUANTIZE",
        "scale_format": scale_format,
        "scale_method": scale_method.name,
        "allowlist": {"types": [], "names": []},
        "blocklist": {"types": [], "names": []},
        "dump_stats_path": f"{temp_directory}/inc_output",
    }
    measure_config = FP8Config.from_dict(measure_config_dict)
    quant_config = FP8Config.from_dict(quant_config_dict)

    def run_convert():
        convert(inference_model, quant_config)

    is_runtime_scale_patching.cache_clear()
    os.environ["RUNTIME_SCALE_PATCHING"] = "0"

    model = prepare(model, measure_config)
    input = torch.tensor([1.2, 2.1]).to(torch.bfloat16).to("hpu")
    model(input)
    finalize_calibration(model)

    if scale_method == ScaleMethodString.ACT_MAXABS_PCS_POW2_WEIGHT_MAXABS_PTS_POW2_HW:
        return run_with_raised_exception(run_convert, ValueError, "Unsupported config: scale_method")
    if dynamic_scale_patching:
        os.environ["RUNTIME_SCALE_PATCHING"] = "1"
        if scale_method not in RUNTIME_SCALE_PATCHING_SUPPORTED_METHODS_LIST:
            run_with_raised_exception(run_convert, AssertionError, "Cannot set scaling attributes.")
            return
    # The following convert should run successfully without any asserts
    inference_model = convert(inference_model, quant_config)
