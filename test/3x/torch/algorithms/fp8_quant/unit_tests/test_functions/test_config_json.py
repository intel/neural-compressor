"""Use this module as an example of how to write new unit tests for layers."""

import os

import pytest
import torch

import neural_compressor.torch.algorithms.fp8_quant as fp8_quant
from neural_compressor.torch.algorithms.fp8_quant._quant_common.helper_modules import Matmul
from neural_compressor.torch.algorithms.fp8_quant._quant_common.quant_config import QuantMode, ScaleMethod

from ...tester import SCALE_METHODS_KEY_ERROR, SCALE_METHODS_QUANT_ONLY, _get_test_only_config


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
@pytest.mark.parametrize("scale_method", ScaleMethod)
@pytest.mark.parametrize("quant_mode", QuantMode)
def test_predefined_config(lp_dtype, scale_method, quant_mode):
    def run_predefined_config():
        config = _get_test_only_config(
            mode=quant_mode,
            lp_dtype=lp_dtype,
            scale_method=scale_method,
        )
        model = Model()
        import neural_compressor.torch.algorithms.fp8_quant.prepare_quant.prepare_model as prepare_model

        prepare_model._prep_model_with_predefined_config(model, config=config)
        fp8_quant.finish_measurements(model)

    def run_with_raises(error, error_str):
        with pytest.raises(Exception) as exc:
            run_predefined_config()
        assert error_str in str(exc.value)
        assert exc.type == error

    # TODO [SW-196641]: fix the following issue:
    if scale_method in SCALE_METHODS_KEY_ERROR and quant_mode == QuantMode.QUANTIZE:
        run_with_raises(KeyError, "(<ScaleMethod.")
    # This is an expected exception, quant only methods support only quantization
    elif scale_method in SCALE_METHODS_QUANT_ONLY and quant_mode != QuantMode.QUANTIZE:
        run_with_raises(ValueError, "Unexpected behavior. This scale method doesn't require measurements.")
    # This is an expected exception, as test is not measuring before
    elif scale_method not in SCALE_METHODS_QUANT_ONLY:
        if quant_mode == QuantMode.QUANTIZE:
            run_with_raises(FileNotFoundError, "Failed to load file ")
        # TODO [SW-196641]: fix the following issue:
        elif quant_mode == QuantMode.SHAPE:
            run_with_raises(UnboundLocalError, "local variable 'fname_base' referenced before assignment")
    else:
        run_predefined_config()
