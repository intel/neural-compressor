# Copyright (c) 2024 Intel Corporation
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

import torch
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.utils.experimental as htexp
from .common import ModuleConfig
from .quant_dequant import cast_to_fp8_fcn, cast_fcn, descale_fcn, scale_fcn
from neural_compressor.torch.utils.auto_accelerator import auto_detect_accelerator
cur_accelerator = auto_detect_accelerator()

GAUDI2 = htexp.synDeviceType.synDeviceGaudi2
GAUDI3 = htexp.synDeviceType.synDeviceGaudi3

EXP_WIDTH = {
    torch.float32: 8,
    torch.bfloat16: 8,
    torch.float8_e4m3fn: 4,
    torch.float8_e5m2: 5,
}


def get_default_exp_bias(dtype):
    exp_width = EXP_WIDTH[dtype]
    return 2 ** (exp_width - 1) - 1


EXP_BIAS_SETS = {
    (GAUDI2, torch.float8_e4m3fn): [3, 7, 11, 15],
    (GAUDI2, torch.float8_e5m2): [15],
    (GAUDI3, torch.float8_e4m3fn): range(0, 63),
    (GAUDI3, torch.float8_e5m2): range(0, 63),
}

MAX_RANGE = {
    torch.float32: torch.finfo(torch.float32).max,
    torch.bfloat16: torch.finfo(torch.bfloat16).max,
    torch.float8_e4m3fn: torch.finfo(torch.float8_e4m3fn).max,
    # float8_e4m3fn data type is 8-bit floating point consist of Exponent: 4, Mantissa: 3, bias: 7. It's supported by Gaudi3.
    torch.float8_e5m2: torch.finfo(torch.float8_e5m2).max,
    # float8_e5m2 data type is 8-bit floating point consist of Exponent: 5, Mantissa: 2, bias: 15. IEEE 754, with NaN and inf.
    torch.float8_e4m3fnuz: torch.finfo(torch.float8_e4m3fnuz).max,
    # float8_e4m3fnuz data type is 8-bit floating point consist of Exponent: 4, Mantissa: 3, bias: 8 with 1 sign bit. It's supported by Gaudi2.
}


def get_fullscale(dtype, device, exp_bias=None):
    default_exp_bias = get_default_exp_bias(dtype)
    if device == GAUDI2 and dtype == torch.float8_e4m3fn:
        fullscale = MAX_RANGE[torch.float8_e4m3fnuz]
    else:
        fullscale = MAX_RANGE[dtype]
    exp_bias = default_exp_bias if exp_bias is None else exp_bias
    fullscale = fullscale * (2 ** (default_exp_bias - exp_bias))
    return fullscale


def get_fullscales_by_expbias_set(dtype, device, expbias_set):
    return [get_fullscale(dtype, device, exp_bias=eb) for eb in expbias_set]


def get_fp8_hw_alligned_scales(dtype, device):
    exp_bias_set = EXP_BIAS_SETS.get((device, dtype), None)
    return (
        None
        if exp_bias_set is None
        else [x / get_fullscale(dtype, device) for x in get_fullscales_by_expbias_set(dtype, device, exp_bias_set)]
    )


DEVICES_SCALE_FACTORS = {
    htexp.synDeviceType.synDeviceGaudi2: 4,
    htexp.synDeviceType.synDeviceGaudi3: 1,
}
FP8_143_SCALES = {
    device: get_fp8_hw_alligned_scales(torch.float8_e4m3fn, device) for device in DEVICES_SCALE_FACTORS.keys()
}
FP8_143_SCALES_TRAITS = {
    device: (
        min(FP8_143_SCALES[device]),
        max(FP8_143_SCALES[device]),
        DEVICES_SCALE_FACTORS[device],
    )
    for device in DEVICES_SCALE_FACTORS.keys()
}


def calc_maxabs_scale(xmaxabs, fullscale, backoff=1):
    scale = xmaxabs / (fullscale * backoff)
    return scale


def scale_to_pow2(scale):
    scale_pow2 = 2 ** torch.ceil(torch.log2(scale))
    return scale_pow2


# Considering range of hw aligned scales: 2^a, 2^a+1,..., 2^b (a<b)
# we want to choose scale s for maxabs m such that 2^a <= s=2^x <= 2^b (for integer a<=x<=b)
# and also 2^(x-1) < m <= 2^x
# if m>=2^b then s=2^b, therefore min(_, 2^b)
# if m<=2^a then s=2^a, therefore max(_, 2^a) --> 2^a <= min(max(_,2^a),2^b) <=2^b
# if s^a<m<2^b then m as a positive number can be written as m=2^y (y=log2(m))
# if y is integer then y=ciel(y) we choose x=y so s=2^x=2^y=2^ciel(y)=2^ciel(log2(m))
# else we choose x=ciel(y) and a<=x-1<y<x<=b and s=2^x=2^ciel(y)=2^ciel(log2(m))
# for Gaudi2 the range is 16^-2..16^1 so we change 2 with 16 and remember that:
# 16 = 2^4, log16(m)=log2(m)/log2(16)=log2(m)/4, and we get:
# we choose s=16^ciel(log16(m))=2^4^ciel(log2(m)/4)=2^(4*ciel(log2(m)/4))=2^(ciel(log2(m)/4)*4)
def scale_to_pow2_hw(scale, device_for_scales):
    scale_pow2 = scale_to_pow2(scale)
    min_scale, max_scale, scale_factor = FP8_143_SCALES_TRAITS[device_for_scales]
    scale_pow2_hw = torch.minimum(
        torch.maximum(
            2 ** (torch.ceil(torch.log2(scale_pow2) / scale_factor) * scale_factor),
            torch.tensor(min_scale, dtype=scale.dtype, device=scale.device),
        ),
        torch.tensor(max_scale, dtype=scale.dtype, device=scale.device),
    )
    return scale_pow2_hw


def mmse_scale_multi(x, ref_scale, scales, lp_dtype, hp_dtype):
    # TODO: SW-176672 move weights to hpu before the scale calculations
    x = x.to("hpu")
    Nch = x.shape[-1]
    opt_err = torch.ones(Nch, dtype=hp_dtype, device=x.device) * torch.inf
    opt_scale = torch.ones(Nch, dtype=hp_dtype, device=x.device) * -1
    sum_axis = list(range(x.ndim - 1))
    rs = ref_scale.unsqueeze(dim=1).transpose(0, 1)
    for s in scales:
        sv = torch.ones(Nch, dtype=hp_dtype, device=x.device) * s
        xscales = rs * sv
        y = scale_fcn(x, xscales)
        y = cast_to_fp8_fcn(y, lp_dtype)
        cur_accelerator.synchronize()  # we are measuring the error so we want to avoid fusion of the converts
        y = cast_fcn(y, hp_dtype)
        y = descale_fcn(y, xscales)
        err = torch.sum((x - y) ** 2, dim=sum_axis)
        opt_scale = torch.where(err < opt_err, sv, opt_scale)
        opt_err = torch.where(err < opt_err, err, opt_err)
        cur_accelerator.synchronize()
    return opt_scale * ref_scale


def mmse_scale(x, scales, lp_dtype, hp_dtype):
    # TODO: SW-176672 move weights to hpu before the scale calculations
    x = x.to("hpu")
    opt_err = torch.ones(1, dtype=hp_dtype, device=x.device) * torch.inf
    opt_scale = -1
    for s in scales:
        y = scale_fcn(x, s)
        y = cast_to_fp8_fcn(y, lp_dtype)
        cur_accelerator.synchronize()  # we are measuring the error so we want to avoid fusion of the converts
        y = cast_fcn(y, hp_dtype)
        y = descale_fcn(y, s)
        err = torch.norm(x - y)
        opt_scale = torch.where(err <= opt_err, s, opt_scale)
        opt_err = torch.where(err <= opt_err, err, opt_err)
        cur_accelerator.synchronize()
    return opt_scale


def manipulate_scales(scales, func):
    new_inputs = [func(input) for input in scales.inputs]
    new_outputs = [func(output) for output in scales.outputs]
    new_weights = {}
    if "weight" in scales.params.keys():
        if isinstance(scales.params["weight"], (torch.Tensor, float)):
            new_weights = {"weight": func(scales.params["weight"])}
        elif isinstance(scales.params["weight"], dict):
            new_weights_dict = {}
            for key, wt in scales.params["weight"].items():
                new_weights_dict[key] = func(wt)
            new_weights = {"weight": new_weights_dict}
    new_scales = ModuleConfig(new_inputs, new_outputs, new_weights)
    return new_scales


def invert_scales(scales):
    def invert(x):
        if isinstance(x, (list, tuple)):
            return [1 / x_i for x_i in x]
        return 1 / x

    return manipulate_scales(scales, invert)
