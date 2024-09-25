import typing

import pytest
import torch

from neural_compressor.torch.algorithms.fp8_quant._quant_common.quant_config import ScaleMethod

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
@pytest.mark.parametrize("scale_method", ScaleMethod)
def test_conv2d_accuracy(hp_dtype: torch.dtype, lp_dtype: torch.dtype, scale_method: ScaleMethod):
    # TODO [SW-196641]: fix the following issues:
    if scale_method in SCALE_METHODS_SEGFAULT:
        pytest.skip("Not supported")
    if scale_method in SCALE_METHODS_KEY_ERROR:
        pytest.xfail("KeyError")
    if scale_method in SCALE_METHODS_COMPILATION_ERROR:
        pytest.xfail("Graph compile error")
    quant_modes = QUANT_MODES_DEFAULT
    atol = 0.2
    if scale_method in SCALE_METHODS_QUANT_ONLY:
        quant_modes = QUANT_MODES_QUANT_ONLY
        if scale_method == ScaleMethod.HW_ALIGNED_SINGLE_SCALE:
            atol = 1.0
    C_in = 1
    C_out = 1
    K = 3

    H = W = 8

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
    )
