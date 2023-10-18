import os
import torch
import habana_frameworks.torch.hpex
from habana_frameworks.torch import _hpex_C as tex
from torch.nn import functional as F
from neural_compressor.common import logger


_F_linear = F.linear
_torch_matmul = torch.matmul


def fp8_linear_forward(input, weight, bias):
    dtype_amax = 240 if os.getenv('PT_USE_FP8_143') is not None else 57344
    use_amax = True
    out_dtype = torch.float32
    logger.debug(f"F.linear dtype_amax: {dtype_amax}, use_amax: {use_amax}")
    input_raw = input
    input = input.view((-1, weight.shape[-1]))
    if use_amax:
        input_scale = dtype_amax / input.abs().max()
        weight_scale = dtype_amax / weight.abs().max()
    else:
        input_scale = torch.tensor(1).to('hpu')
        weight_scale = torch.tensor(1).to('hpu')
    input_scale_inv = 1.0 / input_scale
    weight_scale_inv = 1.0 / weight_scale
    input = torch.ops.hpu.cast_to_fp8_v2(input, input_scale_inv, False, False)[0]
    weight = torch.ops.hpu.cast_to_fp8_v2(weight, weight_scale_inv, False, False)[0]
    out = torch.ops.hpu.fp8_gemm_v2(
        input,
        False,
        weight,
        True,
        None,
        out_dtype,
        input_scale_inv, # inv is used for recover scale
        weight_scale_inv,
        bias,
        False,
    )
    return out.view(-1, *input.shape[1:-1], out.shape[-1])


def fp8_matmul(input1, input2):
    dtype_amax = 240 if os.getenv('PT_USE_FP8_143') is not None else 57344
    use_amax = True
    logger.debug(f"torch.matmul dtype_amax: {dtype_amax}, use_amax: {use_amax}")
    if use_amax:
        input1_scale = dtype_amax / input1.data.abs().max()
        input2_scale = dtype_amax / input2.data.abs().max()
    else:
        input1_scale = torch.tensor(1).to('hpu')
        input2_scale = torch.tensor(1).to('hpu')
    input1_scale_inv = 1.0 / input1_scale
    input2_scale_inv = 1.0 / input2_scale
    input1 = torch.ops.hpu.cast_to_fp8_v2(input1, input1_scale, False, False)[0]
    input2 = torch.ops.hpu.cast_to_fp8_v2(input2, input2_scale, False, False)[0]
    out = torch.ops.hpu.fp8_gemm_v2(
        input1,
        False,
        input2,
        False,
        None,
        torch.float32,
        input1_scale_inv, # inv is used for recover scale
        input2_scale_inv,
        None,
        False,
    )
    return out


def replace_func():
    torch.matmul = fp8_matmul
    F.linear = fp8_linear_forward
    logger.debug("F.linear and torch.matmul are replaced with the fp8 one")


def recover_func():
    F.linear = _F_linear
    torch.matmul = _torch_matmul
    logger.debug("F.linear and torch.matmul are recovered")
