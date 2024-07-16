import typing

import pytest
import torch
from habana_quantization_toolkit._quant_common.quant_config import ScaleMethod
from habana_quantization_toolkit.tests import TestVector, run_accuracy_test


def get_test_vectors(*, dtype: torch.dtype, C_in: int, H: int, W: int) -> typing.Iterable[TestVector]:
    yield TestVector(
        inputs=[torch.ones(1, C_in, H, W, dtype=dtype, device="hpu")],
        atol=0.2,
    )


@pytest.mark.parametrize("hp_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("lp_dtype", [torch.float8_e4m3fn])
def test_conv2d_accuracy(hp_dtype: torch.dtype, lp_dtype: torch.dtype):
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
        scale_method=ScaleMethod.MAXABS_HW,
        test_vectors=get_test_vectors(dtype=hp_dtype, C_in=C_in, H=H, W=W),
    )
