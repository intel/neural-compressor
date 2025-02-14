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

import torch.nn as nn
import torch
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib
from abc import abstractmethod
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.utils.experimental as htexp
from neural_compressor.torch.utils.auto_accelerator import auto_detect_accelerator

cur_accelerator = auto_detect_accelerator()

from .._core.scale_handler import add_scale_registry, get_scale_dtype
from .._quant_common.quant_config import ScaleFormat
from .common import QuantTensorType
from .fp_utils import (
    quantize_per_tensor_to_fp8,
    dequantize_per_tensor_from_fp8,
    quantize_per_channel_to_fp8,
    dequantize_per_channel_from_fp8,
)
from .scale_handler import create_scale_tensor


class QuantDequantBase(nn.Module):
    def __init__(self, lp_dtype, hp_dtype="", *args, **kwargs):
        super().__init__()  # Initialize nn.Module
        add_scale_registry(self)
        self.lp_dtype = lp_dtype
        self.hp_dtype = hp_dtype
        self.scale_format = kwargs.get("scale_format", ScaleFormat.CONST)
        self.use_qdq = kwargs.get("use_qdq", False)
        if self.use_qdq:
            self.qdq_init()

    def qdq_init(self):
        if htexp._get_device_type() == htexp.synDeviceType.synDeviceGaudi2 and self.lp_dtype == torch.float8_e4m3fn:
            self.quant_min = int(torch.finfo(torch.float8_e4m3fnuz).min)
            self.quant_max = int(torch.finfo(torch.float8_e4m3fnuz).max)
        else:
            self.quant_min = int(torch.finfo(self.lp_dtype).min)
            self.quant_max = int(torch.finfo(self.lp_dtype).max)

        if self.scale_format == ScaleFormat.CONST:
            self.zero_point = nn.Parameter(torch.tensor(0.))
        else:
            self.zero_point = 0
        self.forward = self.forward_qdq

    def set_cast_to_op(self):
        return torch.ops.hpu.cast_to_fp8_v2.scalar if self.scale_format == ScaleFormat.SCALAR else \
               torch.ops.hpu.cast_to_fp8_v2

    def set_cast_from_op(self):
        return torch.ops.hpu.cast_from_fp8.scalar if self.scale_format == ScaleFormat.SCALAR else \
               torch.ops.hpu.cast_from_fp8

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def forward_qdq(self, *args, **kwargs):
        pass

    def extra_repr(self) -> str:
        return f"lp_dtype={self.lp_dtype}, hp_dtype={self.hp_dtype}"


class QuantDequantNone(QuantDequantBase):
    def __init__(self, lp_dtype, hp_dtype, *args, **kwargs):
        super(QuantDequantNone, self).__init__(lp_dtype, hp_dtype, *args, **kwargs)

    def forward(self, *args, **kwargs):
        return args[0]

    def extra_repr(self) -> str:
        repr = super(QuantDequantNone, self).extra_repr()
        return f"{repr}, doesn't quantize nor dequantize"


class QuantInput(QuantDequantBase):
    def __init__(self, scale_inv, lp_dtype, hp_dtype, *args, **kwargs):
        super(QuantInput, self).__init__(lp_dtype, hp_dtype, *args, **kwargs)
        scale_inv = scale_inv.unsqueeze(1) if (scale_inv.numel() > 1 and not self.use_qdq) else scale_inv
        self.register_scale("scale_inv", scale_inv, self.scale_format)
        if self.use_qdq:
            self.register_scale("scale", 1 / self.scale_inv, self.scale_format)
            self.quantize_op = (
                quantize_per_channel_to_fp8
                if self.scale_format == ScaleFormat.CONST and self.scale.numel() > 1
                else quantize_per_tensor_to_fp8
            )

        self.cast_to_op = self.set_cast_to_op()

    def forward(self, x):
        return self.cast_to_op(x, self.scale_inv, False, False, self.lp_dtype)[0]

    def forward_qdq(self, x):
        return self.quantize_op(
                x,
                scale=self.scale,
                zero_point=self.zero_point,
                axis=0,
                quant_min=self.quant_min,
                quant_max=self.quant_max,
                dtype=self.lp_dtype,
            )

    def extra_repr(self) -> str:
        repr = super(QuantInput, self).extra_repr()
        dtype = get_scale_dtype(self.scale_inv)
        return f"{repr}, scale_inv dtype={dtype}"


class QuantDynamicInput(QuantDequantBase):
    def __init__(self, input_scales_creator, lp_dtype, hp_dtype, *args, **kwargs):
        super(QuantDynamicInput, self).__init__(lp_dtype, hp_dtype, *args, **kwargs)
        self.input_scales_creator = input_scales_creator

        self.cast_to_op = self.set_cast_to_op()

    def forward(self, x):
        scale = self.input_scales_creator.calc_scales(x, QuantTensorType.DYNAMIC)
        scale_inv = self.input_scales_creator.calc_invert_scales()

        scale = create_scale_tensor(scale, self.scale_format)
        scale_inv = create_scale_tensor(scale_inv, self.scale_format)

        ret = self.cast_to_op(x, scale_inv, False, False, self.lp_dtype)[0]

        return ret, scale

    def extra_repr(self) -> str:
        repr = super(QuantDynamicInput, self).extra_repr()
        return f"{repr} input_scales_creator={self.input_scales_creator}"


class DequantOutput(QuantDequantBase):
    def __init__(self, scale, lp_dtype, hp_dtype, *args, **kwargs):
        super(DequantOutput, self).__init__(lp_dtype, hp_dtype, *args, **kwargs)
        self.register_scale("scale", scale, self.scale_format)
        if self.use_qdq:
            self.dequantize_op = (
                dequantize_per_channel_from_fp8
                if self.scale_format == ScaleFormat.CONST and self.scale.numel() > 1
                else dequantize_per_tensor_from_fp8
            )

        self.cast_from_op = self.set_cast_from_op()

    def forward(self, x):
        return self.cast_from_op(x, self.scale, self.hp_dtype)

    def forward_qdq(self, x):
        return self.dequantize_op(
                x,
                scale= self.scale,
                zero_point=self.zero_point,
                axis=1,
                quant_min=self.quant_min,
                quant_max=self.quant_max,
                dtype=self.lp_dtype,
                out_dtype=self.hp_dtype,
            )

    def extra_repr(self) -> str:
        repr = super(DequantOutput, self).extra_repr()
        dtype = get_scale_dtype(self.scale)
        return f"{repr}, scale dtype={dtype}"


class QuantDequant(QuantDequantBase):
    def __init__(self, scale_inv, lp_dtype, hp_dtype, *args, **kwargs):
        super(QuantDequant, self).__init__(lp_dtype, hp_dtype, *args, **kwargs)
        self.register_scale("scale_inv", scale_inv, self.scale_format)
        self.register_scale("scale", 1 / scale_inv, self.scale_format)
        self.cast_to_op = self.set_cast_to_op()
        self.cast_from_op = self.set_cast_from_op()

    def forward(self, x, *args, **kwargs):
        y = self.cast_to_op(x, self.scale_inv, False, False, self.lp_dtype)[0]
        # mark_step is needed so fuser won't remove 2 consecutive casts.
        # will be removed once SW-196431 is implemented
        # Call cur_accelerator.synchronize() which will call mark_step() as well
        cur_accelerator.synchronize()
        z = self.cast_from_op(y, self.scale, self.hp_dtype)
        cur_accelerator.synchronize()
        return z

    def forward_qdq(self, x, *args, **kwargs):
        y = quantize_per_tensor_to_fp8(
            x,
            scale=self.scale,
            zero_point=self.zero_point,
            quant_min=self.quant_min,
            quant_max=self.quant_max,
            dtype=self.lp_dtype,
        )
        z = dequantize_per_tensor_from_fp8(
            y,
            scale=self.scale,
            zero_point=self.zero_point,
            quant_min=self.quant_min,
            quant_max=self.quant_max,
            dtype=self.lp_dtype,
            out_dtype=self.hp_dtype,
        )
        return z

    def extra_repr(self) -> str:
        repr = super(QuantDequant, self).extra_repr()
        return f"{repr}, Quantize, and then dequantize"
