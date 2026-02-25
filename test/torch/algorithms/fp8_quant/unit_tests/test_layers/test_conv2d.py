import typing

import pytest
import torch

from neural_compressor.torch.algorithms.fp8_quant._core.scale_methods.scale_method_config import ScaleMethodString

from ...test_hpu_utils import *
from ...tester import *


def get_test_vectors(
    *, dtype: torch.dtype, C_in: int, H: int, W: int, atol: float = 0.2
) -> typing.Iterable[TestVector]:
    yield TestVector(
        inputs=[torch.ones(1, C_in, H, W, dtype=dtype, device="hpu")],
        atol=atol,
    )


@pytest.mark.parametrize("hp_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("lp_dtype", [torch.float8_e4m3fn], ids=["fp8_e4m3fn"])
@pytest.mark.parametrize("scale_method", ScaleMethodString)
@pytest.mark.parametrize("device_type", device_type)
def test_conv2d_accuracy(hp_dtype: torch.dtype, lp_dtype: torch.dtype, scale_method: ScaleMethodString, device_type: str):
    # TODO [SW-196641]: fix the following issues:
    if scale_method in SCALE_METHODS_SEGFAULT:
        pytest.skip("Not supported")
    if scale_method in SCALE_METHODS_COMPILATION_ERROR:
        pytest.xfail("Graph compile error")
    quant_modes = QUANT_MODES_DEFAULT
    atol = 0.2
    if scale_method in SCALE_METHODS_QUANT_ONLY:
        quant_modes = QUANT_MODES_QUANT_ONLY
        if scale_method == ScaleMethodString.HW_ALIGNED_SINGLE_SCALE:
            atol = 1.0
    C_in = 1
    C_out = 1
    K = 3

    H = W = 8

    def run():
        run_accuracy_test(
            module_class=torch.nn.Conv2d,
            module_kwargs={
                "in_channels": C_in,
                "out_channels": C_out,
                "kernel_size": K,
                "padding": 1,
                "bias": False,
                "device": "hpu",
                "dtype": hp_dtype,
            },
            lp_dtype=lp_dtype,
            scale_method=scale_method,
            test_vectors=get_test_vectors(dtype=hp_dtype, C_in=C_in, H=H, W=W, atol=atol),
            quant_modes=quant_modes,
            device_type=device_type,
        )
    if get_device_type() != device_type_id[device_type] and scale_method != ScaleMethodString.MAXABS_HW:
        return run_with_raised_exception(run, ValueError, "Unsupported config: scale_method")
    elif device_type_id[device_type] != get_device_type():
        if not (device_type_id[device_type] == get_gaudi2_type() and is_gaudi3()):
            return run_with_raised_exception(run, ValueError, "Unsupported config: device_for_scales=")
    elif scale_method == ScaleMethodString.ACT_MAXABS_PCS_POW2_WEIGHT_MAXABS_PTS_POW2_HW:
            return run_with_raised_exception(run, ValueError, "Unsupported config: scale_method ACT_MAXABS_PCS_POW2_WEIGHT_MAXABS_PTS_POW2_HW")
    return run()
