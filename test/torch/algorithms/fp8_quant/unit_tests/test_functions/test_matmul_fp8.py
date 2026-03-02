import itertools
from typing import Iterable, Tuple

import habana_frameworks.torch.utils.experimental as htexp
import pytest
import torch

from neural_compressor.torch.algorithms.fp8_quant._core.fp_utils import FP8_143_SCALES
from neural_compressor.torch.algorithms.fp8_quant._core.quantized_func_wrappers.hpu.hpu_quantized_func_wrapper import (
    QuantizedHpuMatmul,
)
from neural_compressor.torch.algorithms.fp8_quant._quant_common.quant_config import ScaleFormat
from neural_compressor.torch.utils.auto_accelerator import auto_detect_accelerator


def run_test_matmul_fp8(
    *,
    hp_dtype: torch.dtype,
    lp_dtype: torch.dtype,
    scales: Tuple[float, float],
):
    torch.manual_seed(0)
    x = torch.randn(2, 2, dtype=float).clone()
    y = torch.randn(2, 2, dtype=float).clone()

    x_scale, y_scale = scales
    expected_result = (torch.matmul(x, y) / x_scale / y_scale).to(dtype=hp_dtype)
    matmul_fp8 = QuantizedHpuMatmul(ScaleFormat.SCALAR)
    result = matmul_fp8(
        input=x.to(device="hpu").to(dtype=lp_dtype),
        other=y.to(device="hpu").to(dtype=lp_dtype),
        out_dtype=hp_dtype,
        scale_input_inv=1 / x_scale,
        scale_other_inv=1 / y_scale,
    )

    assert torch.allclose(expected_result, result, rtol=0.1), f"Matmul failed for {x_scale=} {y_scale=}"


def get_fp8_143_scales():
    inc_device_type = auto_detect_accelerator().get_inc_accelerator_type()
    return FP8_143_SCALES[inc_device_type]


def get_scales_pairs_not_both_hw_aligned() -> Iterable[Tuple[float, float]]:
    not_hw_aligned_scales = [0.25]

    return itertools.chain(
        zip(not_hw_aligned_scales, not_hw_aligned_scales),
        zip(not_hw_aligned_scales, get_fp8_143_scales()),
        zip(get_fp8_143_scales(), not_hw_aligned_scales),
    )


def get_scales_pairs_both_hw_aligned() -> Iterable[Tuple[float, float]]:
    return zip(get_fp8_143_scales(), get_fp8_143_scales())


@pytest.mark.parametrize("hp_dtype", [torch.bfloat16])
@pytest.mark.parametrize("lp_dtype", [torch.float8_e4m3fn])
def test_matmul_fp8_not_both_hw_aligned(
    hp_dtype: torch.dtype,
    lp_dtype: torch.dtype,
):
    for scales in get_scales_pairs_not_both_hw_aligned():
        run_test_matmul_fp8(hp_dtype=hp_dtype, lp_dtype=lp_dtype, scales=scales)


@pytest.mark.parametrize("hp_dtype", [torch.bfloat16])
@pytest.mark.parametrize("lp_dtype", [torch.float8_e4m3fn])
def test_matmul_fp8_both_hw_aligned(
    hp_dtype: torch.dtype,
    lp_dtype: torch.dtype,
):
    for scales in get_scales_pairs_both_hw_aligned():
        run_test_matmul_fp8(hp_dtype=hp_dtype, lp_dtype=lp_dtype, scales=scales)
