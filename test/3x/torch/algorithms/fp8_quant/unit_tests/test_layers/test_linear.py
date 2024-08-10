import typing

import pytest
import torch
from habana_quantization_toolkit._quant_common.quant_config import ScaleMethod
from habana_quantization_toolkit.tests import TestVector, run_accuracy_test


def get_test_vectors(*, dtype: torch.dtype, N: int, D_in: int) -> typing.Iterable[TestVector]:
    yield TestVector(
        inputs=[torch.ones(N, D_in, dtype=dtype, device="hpu")],
        atol=0.02,
    )


@pytest.mark.parametrize("hp_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("lp_dtype", [torch.float8_e4m3fn])
def test_linear_accuracy(hp_dtype: torch.dtype, lp_dtype: torch.dtype):
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
        scale_method=ScaleMethod.MAXABS_HW,
        test_vectors=get_test_vectors(dtype=hp_dtype, N=N, D_in=D_in),
    )
