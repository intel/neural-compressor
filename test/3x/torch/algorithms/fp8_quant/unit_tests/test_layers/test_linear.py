import typing

import pytest
import torch

from neural_compressor.torch.algorithms.fp8_quant._quant_common.quant_config import ScaleMethod, ScaleFormat

from ...test_hpu_utils import *
from ...tester import *


def get_test_vectors(*, dtype: torch.dtype, N: int, D_in: int, atol: float = 0.02) -> typing.Iterable[TestVector]:
    yield TestVector(
        inputs=[torch.ones(N, D_in, dtype=dtype, device="hpu")],
        atol=atol,
    )


@pytest.mark.parametrize("hp_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("lp_dtype", [torch.float8_e4m3fn], ids=["fp8_e4m3fn"])
@pytest.mark.parametrize("scale_method", ScaleMethod)
@pytest.mark.parametrize("device_type", device_type)
@pytest.mark.parametrize("scale_format", ScaleFormat)
def test_linear_accuracy(
    hp_dtype: torch.dtype, lp_dtype: torch.dtype, scale_method: ScaleMethod, device_type: str, scale_format: ScaleFormat
):
    if scale_method in SCALE_METHODS_KEY_ERROR:
        pytest.xfail("KeyError")
    # TODO [SW-215692]: Fix segfault
    if scale_format == ScaleFormat.CONST:
        if scale_method in [ScaleMethod.MAXABS_HW_OPT_WEIGHT, ScaleMethod.MAXABS_POW2_OPT_WEIGHT]:
            pytest.xfail("Segfault")
    quant_modes = QUANT_MODES_DEFAULT
    atol = 0.022
    if scale_method == ScaleMethod.MAXABS_ARBITRARY:
        atol = 0.03
    if scale_method in SCALE_METHODS_QUANT_ONLY:
        quant_modes = QUANT_MODES_QUANT_ONLY
        if scale_method == ScaleMethod.HW_ALIGNED_SINGLE_SCALE:
            atol = 1.0
    N = 1
    D_in = 8
    H = 5
    def run():
        run_accuracy_test(
            module_class=torch.nn.Linear,
            module_kwargs={
                "in_features": D_in,
                "out_features": H,
                "bias": False,
                "dtype": hp_dtype,
            },
            lp_dtype=lp_dtype,
            scale_method=scale_method,
            test_vectors=get_test_vectors(dtype=hp_dtype, N=N, D_in=D_in, atol=atol),
            quant_modes=quant_modes,
            device_type=device_type,
            scale_format=scale_format,
        )
    if get_device_type() != device_type_id[device_type] and scale_method != ScaleMethod.MAXABS_HW:
        return run_with_raised_exception(run, ValueError, "Unsupported config: scale_method: ")
    elif device_type_id[device_type] != get_device_type():
        if not (device_type_id[device_type] == get_gaudi2_type() and is_gaudi3()):
            return run_with_raised_exception(run, ValueError, "Unsupported config: device_for_scales=")
    return run()
