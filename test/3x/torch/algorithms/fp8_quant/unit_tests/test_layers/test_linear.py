import typing

import pytest
import torch

from neural_compressor.torch.algorithms.fp8_quant._quant_common.quant_config import ScaleMethod

from ...tester import *


def get_test_vectors(*, dtype: torch.dtype, N: int, D_in: int, atol: float = 0.02) -> typing.Iterable[TestVector]:
    yield TestVector(
        inputs=[torch.ones(N, D_in, dtype=dtype, device="hpu")],
        atol=atol,
    )


@pytest.mark.parametrize("hp_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("lp_dtype", [torch.float8_e4m3fn], ids=["fp8_e4m3fn"])
@pytest.mark.parametrize("scale_method", ScaleMethod)
def test_linear_accuracy(hp_dtype: torch.dtype, lp_dtype: torch.dtype, scale_method: ScaleMethod):
    # TODO [SW-196641]: fix the following issues:
    if scale_method in [
        ScaleMethod.ACT_MAXABS_HW_WEIGHTS_PCS_OPT_POW2,
        ScaleMethod.ACT_MAXABS_POW2_WEIGHTS_PCS_OPT_POW2,
        ScaleMethod.MAXABS_HW_OPT_WEIGHT,
        ScaleMethod.MAXABS_POW2_OPT_WEIGHT,
        ScaleMethod.ACT_MAXABS_HW_WEIGHTS_PCS_MAXABS_POW2,
        ScaleMethod.ACT_MAXABS_POW2_WEIGHTS_PCS_MAXABS_POW2,
    ]:
        pytest.skip("Not supported")
    if scale_method in SCALE_METHODS_KEY_ERROR:
        pytest.xfail("KeyError")
    quant_modes = QUANT_MODES_DEFAULT
    atol = 0.02
    if scale_method in SCALE_METHODS_QUANT_ONLY:
        quant_modes = QUANT_MODES_QUANT_ONLY
        if scale_method == ScaleMethod.HW_ALIGNED_SINGLE_SCALE:
            atol = 1.0
    N = 1
    D_in = 8
    H = 5
    run_accuracy_test(
        module_class=torch.nn.Linear,
        module_kwargs={
            "in_features": D_in,
            "out_features": H,
            "bias": False,
        },
        lp_dtype=lp_dtype,
        scale_method=scale_method,
        test_vectors=get_test_vectors(dtype=hp_dtype, N=N, D_in=D_in, atol=atol),
        quant_modes=quant_modes,
    )
