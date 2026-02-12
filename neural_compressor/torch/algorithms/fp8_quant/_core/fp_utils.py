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
from enum import Enum
from .common import ModuleConfig
from neural_compressor.torch.utils.auto_accelerator import auto_detect_accelerator, INCAcceleratorType
from neural_compressor.torch.utils import logger

cur_accelerator = auto_detect_accelerator()

descale_fcn = lambda x, scale: torch.mul(x, scale)
scale_fcn = lambda x, scale: torch.div(x, scale)
cast_fcn = lambda x, dtype: x.to(dtype=dtype)
cast_to_fp8_fcn = lambda x, dtype, scale_inv=None: torch.ops.hpu.cast_to_fp8_v2(x, scale_inv, False, False, dtype)[0]
def calculate_scale_maxabs_with_cguid(x, maxMode, **kwargs):
    return torch.ops.hpu.calculate_scale_for_cast(
        x, maxMode.value, ScaleCalculationRoundingMode.NO_SCALE_ROUNDING.value, **kwargs
    )


def calculate_scale_rounding_with_cguid(x, scaleMode, **kwargs):
    return torch.ops.hpu.calculate_scale_for_cast(
        x, ScaleCalculationMaxMode.NO_MAX_CALCULATION.value, scaleMode.value, **kwargs
    )


class ScaleCalculationMaxMode(Enum):
    NO_MAX_CALCULATION = 0
    MAX_ABS_PTS_CALCULATION = 1
    MAX_ABS_PCS_CALCULATION = 2


class ScaleCalculationRoundingMode(Enum):
    NO_SCALE_ROUNDING = 0
    SCALE_TO_POW2_ROUNDING = 1

GAUDI2 = INCAcceleratorType.GAUDI2
GAUDI3 = INCAcceleratorType.GAUDI3

EXP_WIDTH = {
    torch.float32: 8,
    torch.bfloat16: 8,
    torch.float8_e4m3fn: 4,
    torch.float8_e5m2: 5,
}

descale_fcn = lambda x, scale: torch.mul(x, scale)
scale_fcn = lambda x, scale: torch.div(x, scale)
cast_fcn = lambda x, dtype: x.to(dtype=dtype)
cast_to_fp8_fcn = lambda x, dtype, scale_inv=None: torch.ops.hpu.cast_to_fp8_v2(x, scale_inv, False, False, dtype)[0]
cast_from_fp8_fcn = lambda x, dtype, scale=None: torch.ops.hpu.cast_from_fp8(x, scale, dtype)
quantize_per_tensor_to_fp8 = lambda x, scale, zero_point, quant_min, quant_max, dtype=None, axis=None: torch.ops.quantized_decomposed.quantize_per_tensor(
    x, scale, zero_point, quant_min, quant_max, dtype=dtype
)
dequantize_per_tensor_from_fp8 = lambda x, scale, zero_point, quant_min, quant_max, dtype, out_dtype=None, axis=None: torch.ops.quantized_decomposed.dequantize_per_tensor(
    x, scale, zero_point, quant_min, quant_max, dtype=dtype, out_dtype=out_dtype
)
quantize_per_channel_to_fp8 = lambda x, scale, zero_point, axis, quant_min, quant_max, dtype=None: torch.ops.quantized_decomposed.quantize_per_channel(
    x, scale, zero_point, axis, quant_min, quant_max, dtype=dtype
)
dequantize_per_channel_from_fp8 = lambda x, scale, zero_point, axis, quant_min, quant_max, dtype, out_dtype=None: torch.ops.quantized_decomposed.dequantize_per_channel(
    x, scale, zero_point, axis, quant_min, quant_max, dtype=dtype, out_dtype=out_dtype
)


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
    torch.float8_e5m2: torch.finfo(torch.float8_e5m2).max
    # float8_e5m2 data type is 8-bit floating point consist of Exponent: 5, Mantissa: 2, bias: 15. IEEE 754, with NaN and inf.
}
# TODO FSW-12066 solve fp_utils
try:
    MAX_RANGE[torch.float8_e4m3fnuz] = torch.finfo(torch.float8_e4m3fnuz).max
    # float8_e4m3fnuz data type is 8-bit floating point consist of Exponent: 4, Mantissa: 3, bias: 8 with 1 sign bit. It's supported by Gaudi2.
except AttributeError as e:
    pass

def get_fullscale(dtype, device, exp_bias=None):
    default_exp_bias = get_default_exp_bias(dtype)
    fullscale = 1
    if device == GAUDI2 and dtype == torch.float8_e4m3fn:
        # TODO FSW-12066 solve fp_utils
        try:
            fullscale = MAX_RANGE[torch.float8_e4m3fnuz]
        except AttributeError as e:
            pass
    else:
        fullscale = MAX_RANGE[dtype]
    exp_bias = default_exp_bias if exp_bias is None else exp_bias
    fullscale = fullscale * (2 ** (default_exp_bias - exp_bias))
    return float(fullscale)


def get_fullscales_by_expbias_set(dtype, device, expbias_set):
    return [get_fullscale(dtype, device, exp_bias=eb) for eb in expbias_set]


def get_fp8_hw_alligned_scales_by_device(dtype, device):
    if device not in [GAUDI2, GAUDI3]:
        logger.warning("hw aligned scales not supported for device {}".format(device))
        return None # only Gaudis support hw aligned scales
    exp_bias_set = EXP_BIAS_SETS.get((device, dtype), None)
    return (
        None
        if exp_bias_set is None
        else [x / get_fullscale(dtype, device) for x in get_fullscales_by_expbias_set(dtype, device, exp_bias_set)]
    )

def get_fp8_hw_alligned_scales(dtype):
    inc_device_type = auto_detect_accelerator().get_inc_accelerator_type()
    return get_fp8_hw_alligned_scales_by_device(dtype, inc_device_type)

DEVICES_SCALE_FACTORS = {
    INCAcceleratorType.GAUDI2: 4,
    INCAcceleratorType.GAUDI3: 1,
}
FP8_143_SCALES = {
    device: get_fp8_hw_alligned_scales_by_device(torch.float8_e4m3fn, device) for device in DEVICES_SCALE_FACTORS.keys()
}
FP8_143_SCALES_TRAITS = {
    device: (
        min(FP8_143_SCALES[device]),
        max(FP8_143_SCALES[device]),
        DEVICES_SCALE_FACTORS[device],
    )
    for device in DEVICES_SCALE_FACTORS.keys()
}

def calc_scale_from_maxabs(xmaxabs, fullscale, backoff=1):
    scale = xmaxabs / (fullscale * backoff)
    return scale

def mmse_scale_multi(x, ref_scale, scales, lp_dtype, hp_dtype):
    if not scales:
        raise ValueError(
            "got empty scale list. it is possible that scale method isn't supported by current device."
        )
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
    if not scales:
        raise ValueError(
            "got empty scale list. it is possible that scale method isn't supported by current device."
        )
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
    """Applies a function to the inputs, outputs, and weights of the ModuleConfig object."""
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


def invert_scale(x):
    """Inverts the scale of the input tensor, list of tensors, or tuple of tensors."""
    def invert(x):
        if isinstance(x, torch.Tensor):
            return torch.reciprocal(x)
        return 1.0 / x
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return type(x)(invert(x_i) for x_i in x)
    return invert(x)


def invert_scales(scales):
    """Inverts the scales of the input ModuleConfig object."""
    return manipulate_scales(scales, invert_scale)
