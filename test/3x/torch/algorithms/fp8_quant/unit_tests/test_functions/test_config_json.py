"""Use this module as an example of how to write new unit tests for layers."""

import os
import sys
import pytest
import torch

import neural_compressor.torch.algorithms.fp8_quant as fp8_quant
from neural_compressor.torch.algorithms.fp8_quant._quant_common.helper_modules import Matmul
from neural_compressor.torch.algorithms.fp8_quant._quant_common.quant_config import QuantMode
from neural_compressor.torch.algorithms.fp8_quant._core.scale_methods.scale_method_config import ScaleMethodString
from ...tester import run_with_raised_exception, get_internal_config, SCALE_METHODS_QUANT_ONLY
from ...test_hpu_utils import *

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = Matmul()


def test_config_json():
    model = Model()

    for mode in [QuantMode.MEASURE, QuantMode.QUANTIZE]:
        name = {
            QuantMode.MEASURE: "measure",
            QuantMode.QUANTIZE: "quant",
        }[mode]
        config_path = f"llama_{name}.json"
        # config comes from f"neural_compressor/torch/algorithms/fp8_quant/custom_config/llama_{name}.json"
        fp8_quant.prep_model(model, config_path=config_path)
        fp8_quant.finish_measurements(model)


@pytest.mark.parametrize("lp_dtype", [torch.float8_e4m3fn], ids=["fp8_e4m3fn"])
@pytest.mark.parametrize("scale_method", ScaleMethodString)
@pytest.mark.parametrize("quant_mode", QuantMode)
def test_predefined_config(lp_dtype, scale_method, quant_mode):
    def run_predefined_config():
        config = get_internal_config(
            mode=quant_mode,
            lp_dtype=lp_dtype,
            scale_method=scale_method,
        )
        model = Model()
        print(config)
        import neural_compressor.torch.algorithms.fp8_quant.prepare_quant.prepare_model as prepare_model

        prepare_model._prep_model_with_predefined_config(model, config=config)
        fp8_quant.finish_measurements(model)


    if scale_method == ScaleMethodString.ACT_MAXABS_PCS_POW2_WEIGHT_MAXABS_PTS_POW2_HW:
        return run_with_raised_exception(run_predefined_config, ValueError, "Unsupported config: scale_method")
    # This is an expected exception, as test is not measuring before
    elif scale_method not in SCALE_METHODS_QUANT_ONLY:
        if quant_mode == QuantMode.QUANTIZE:
            run_with_raised_exception(run_predefined_config, FileNotFoundError, "Failed to load file ")
        # TODO [SW-196641]: fix the following issue:
    elif quant_mode == QuantMode.SHAPE:
        error_message = (
            "cannot access local variable 'fname_base' where it is not associated with a value"
            if sys.version_info >= (3, 11)
            else "local variable 'fname_base' referenced before assignment"
        )
        run_with_raised_exception(run_predefined_config, UnboundLocalError, error_message)
    else:
        run_predefined_config()


@pytest.mark.parametrize("lp_dtype", [torch.float8_e4m3fn], ids=["fp8_e4m3fn"])
@pytest.mark.parametrize("quant_mode", QuantMode)
@pytest.mark.parametrize("device_type", device_type)
def test_device_override(lp_dtype, quant_mode, device_type):
    def run_predefined_config():
        config = get_internal_config(
                mode=quant_mode,
                lp_dtype=lp_dtype,
                scale_method=ScaleMethodString.MAXABS_HW,
                device_type=device_type,
            )
        assert config.cfg["device_for_scales"] == htexp_device_type_to_inc_acclerator_type(device_type_id[device_type])
    if device_type_id[device_type] != get_device_type():
        if not (device_type_id[device_type] == get_gaudi2_type() and is_gaudi3()):
            return run_with_raised_exception(run_predefined_config, ValueError, "Unsupported config: device_for_scales=")
    return run_predefined_config()
