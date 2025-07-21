import typing

import pytest
import torch

from neural_compressor.torch.algorithms.fp8_quant._core.scale_methods.scale_method_config import ScaleMethodString

from ...test_hpu_utils import *
from ...tester import *

SUPPORTED_DYNAMIC_SCALES= [ScaleMethodString.ACT_MAXABS_PCS_POW2_WEIGHT_MAXABS_PTS_POW2_HW]


def get_test_vectors(*, dtype: torch.dtype, atol) -> typing.Iterable[TestVector]:
    yield TestVector(
        inputs=[
            torch.eye(2, dtype=dtype, device="hpu"),
            torch.eye(2, dtype=dtype, device="hpu"),
        ],
        atol=atol,
    )
    yield TestVector(
        inputs=[
            torch.randn((2, 2), dtype=dtype, device="hpu"),
            torch.randn((2, 2), dtype=dtype, device="hpu"),
        ],
        atol=atol,
    )
    yield TestVector(
        inputs=[
            torch.eye(2, dtype=dtype, device="hpu"),
            torch.randn((2, 2), dtype=dtype, device="hpu"),
        ],
        atol=atol,
    )


class Matmul(torch.nn.Module):
    """This is a mimic of other implementations of `Matmul`.

    It is here to not create a dependency on optimum-habana (which is logically needed).
    It should not be used directly in user code.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)


@pytest.mark.parametrize("hp_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("lp_dtype", [torch.float8_e4m3fn], ids=["fp8_e4m3fn"])
@pytest.mark.parametrize("scale_method", ScaleMethodString)
@pytest.mark.parametrize("device_type", device_type)
@pytest.mark.parametrize("dynamic_quantization", [True, False], ids=["dynamic_quantization", "static_quantization"])
def test_matmul_accuracy(hp_dtype: torch.dtype, lp_dtype: torch.dtype, scale_method: ScaleMethodString, device_type: str, dynamic_quantization: bool):
    quant_modes = QUANT_MODES_DEFAULT
    atol = 0.2
    if scale_method in SCALE_METHODS_QUANT_ONLY or dynamic_quantization:
        quant_modes = QUANT_MODES_QUANT_ONLY
        if scale_method == ScaleMethodString.HW_ALIGNED_SINGLE_SCALE:
            atol = 1.0
    def run():
        run_accuracy_test(
            module_class=Matmul,
            lp_dtype=lp_dtype,
            scale_method=scale_method,
            test_vectors=get_test_vectors(dtype=hp_dtype, atol=atol),
            quant_modes=quant_modes,
            device_type=device_type,
            dynamic_quantization=dynamic_quantization,
        )

    if scale_method == ScaleMethodString.MAXABS_HW:
        if device_type_id[device_type] == get_gaudi3_type() and is_gaudi2():
            # Gaudi3 scales not supported on Gaudi2 so "device_for_scales:Gaudi3" is not supported on Gaudi2 run
            return run_with_raised_exception(run, ValueError, "Unsupported config: device_for_scales=")
    else:
        if get_device_type() != device_type_id[device_type]:
            # In scale_method different than MAXABS_HW, we don't support device_for_scales so this scale_method config fails
            return run_with_raised_exception(run, ValueError, "Unsupported config: scale_method")

    if dynamic_quantization:
        if scale_method in HW_ALIGNED_SCALE_METHODS or scale_method in QUANT_ONLY_SCALE_METHODS:
            # When in dynamic quantization we don't support hw aligned scale methods and unit scale
            return run_with_raised_exception(run, ValueError, "Unsupported config: scale_method")
    else :
        if scale_method in SUPPORTED_DYNAMIC_SCALES:
            # When in static quantization we don't support dynamic scale method
            return run_with_raised_exception(run, ValueError, "Unsupported config: scale_method")
    return run()
