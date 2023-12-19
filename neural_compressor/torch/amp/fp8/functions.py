# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint:disable=import-error

import os

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpex
import torch
from torch.nn import functional as F

from neural_compressor.common import logger

_F_linear = F.linear
_torch_matmul = torch.matmul
_torch_bmm = torch.bmm


DATA_TYPE = torch.float8_e4m3fn
# without scale factor 0.9, the output will be abnormal.
E4M3_AMAX = torch.tensor(240 * 0.9, dtype=torch.float).to("hpu")
E5M2_AMAX = torch.tensor(57344 * 0.9, dtype=torch.float).to("hpu")

DTYPE_AMAX = E4M3_AMAX if DATA_TYPE == torch.float8_e4m3fn else E5M2_AMAX
USE_AMAX = False if os.getenv("PT_USE_FP8_AMAX") is None else True


def fp8_linear_forward(input, weight, bias):
    out_dtype = torch.float32
    org_middle_shape = input.shape[1:-1]
    input = input.view((-1, weight.shape[-1]))
    # process input
    if input.dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
        out_dtype = input.dtype
        if USE_AMAX:
            input_scale = DTYPE_AMAX / input.data.abs().max()
            input_scale_inv = torch.reciprocal(input_scale)
        else:
            input_scale, input_scale_inv = None, None
        input = torch.ops.hpu.cast_to_fp8_v2(input, input_scale, False, False, DATA_TYPE)[0]
    else:
        # skip cast for input
        input_scale, input_scale_inv = None, None
    # process weight
    if weight.dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
        out_dtype = weight.dtype
        if USE_AMAX:
            weight_scale = DTYPE_AMAX / weight.data.abs().max()
            weight_scale_inv = torch.reciprocal(weight_scale)
        else:
            weight_scale, weight_scale_inv = None, None
        weight = torch.ops.hpu.cast_to_fp8_v2(weight, weight_scale, False, False, DATA_TYPE)[0]
    else:
        # skip cast for weight
        weight_scale, weight_scale_inv = None, None
    out = torch.ops.hpu.fp8_gemm_v2(
        input,
        False,
        weight,
        True,
        None,
        out_dtype,
        input_scale_inv,  # inv is used for recover scale
        weight_scale_inv,
        bias,
        False,
    )
    out = out.view(-1, *org_middle_shape, out.shape[-1])
    return out


def fp8_matmul(input1, input2):
    out_dtype = torch.float32
    # process input1
    if input1.dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
        out_dtype = input1.dtype
        if USE_AMAX:
            input1_scale = DTYPE_AMAX / input1.data.abs().max()
            input1_scale_inv = torch.reciprocal(input1_scale)
        else:
            input1_scale, input1_scale_inv = None, None
        input1 = torch.ops.hpu.cast_to_fp8_v2(input1, input1_scale, False, False, DATA_TYPE)[0]
    else:
        # skip cast for input1
        input1_scale, input1_scale_inv = None, None
    # process input2
    if input2.dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
        out_dtype = input2.dtype
        if USE_AMAX:
            input2_scale = DTYPE_AMAX / input2.data.abs().max()
            input2_scale_inv = torch.reciprocal(input2_scale)
        else:
            input2_scale, input2_scale_inv = None, None
        input2 = torch.ops.hpu.cast_to_fp8_v2(input2, input2_scale, False, False, DATA_TYPE)[0]
    else:
        # skip cast for input2
        input2_scale, input2_scale_inv = None, None
    # calculate
    out = torch.ops.hpu.fp8_gemm_v2(
        input1,
        False,
        input2,
        False,
        None,
        out_dtype,
        input1_scale_inv,  # inv is used for recover scale
        input2_scale_inv,
        None,
        False,
    )
    return out


def replace_func(dtype):
    global DATA_TYPE
    DATA_TYPE = dtype
    F.linear = fp8_linear_forward
    torch.matmul = fp8_matmul
    torch.bmm = fp8_matmul
    logger.debug("F.linear and torch.matmul are replaced with the fp8 one")


def recover_func():
    F.linear = _F_linear
    torch.matmul = _torch_matmul
    torch.bmm = _torch_bmm
    logger.debug("F.linear and torch.matmul are recovered")
