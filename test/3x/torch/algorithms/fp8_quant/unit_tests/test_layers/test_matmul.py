import typing

import pytest
import torch

from neural_compressor.torch.algorithms.fp8_quant._quant_common.quant_config import ScaleMethod

from ...tester import *


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


@pytest.mark.parametrize("hp_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("lp_dtype", [torch.float8_e4m3fn], ids=["fp8_e4m3fn"])
@pytest.mark.parametrize("scale_method", ScaleMethod)
def test_matmul_accuracy(hp_dtype: torch.dtype, lp_dtype: torch.dtype, scale_method: ScaleMethod):
    # TODO [SW-196641]: fix the following issues:
    if scale_method in SCALE_METHODS_KEY_ERROR:
        pytest.xfail("KeyError")
    quant_modes = QUANT_MODES_DEFAULT
    if scale_method in SCALE_METHODS_QUANT_ONLY:
        quant_modes = QUANT_MODES_QUANT_ONLY
    run_accuracy_test(
        module_class=Matmul,
        lp_dtype=lp_dtype,
        scale_method=scale_method,
        test_vectors=get_test_vectors(dtype=hp_dtype),
        quant_modes=quant_modes,
    )
