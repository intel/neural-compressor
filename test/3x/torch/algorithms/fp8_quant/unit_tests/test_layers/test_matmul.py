import typing

import pytest
import torch
from habana_quantization_toolkit._quant_common.quant_config import ScaleMethod
from habana_quantization_toolkit.tests import TestVector, run_accuracy_test


def get_test_vectors(*, dtype: torch.dtype) -> typing.Iterable[TestVector]:
    yield TestVector(
        inputs=[
            torch.eye(2, dtype=dtype, device="hpu"),
            torch.eye(2, dtype=dtype, device="hpu"),
        ],
        atol=0.2,
    )
    yield TestVector(
        inputs=[
            torch.randn((2, 2), dtype=dtype, device="hpu"),
            torch.randn((2, 2), dtype=dtype, device="hpu"),
        ],
        atol=0.2,
    )
    yield TestVector(
        inputs=[
            torch.eye(2, dtype=dtype, device="hpu"),
            torch.randn((2, 2), dtype=dtype, device="hpu"),
        ],
        atol=0.2,
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


@pytest.mark.parametrize("hp_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("lp_dtype", [torch.float8_e4m3fn])
def test_matmul_accuracy(hp_dtype: torch.dtype, lp_dtype: torch.dtype):
    run_accuracy_test(
        module_class=Matmul,
        lp_dtype=lp_dtype,
        scale_method=ScaleMethod.MAXABS_HW,
        test_vectors=get_test_vectors(dtype=hp_dtype),
    )
