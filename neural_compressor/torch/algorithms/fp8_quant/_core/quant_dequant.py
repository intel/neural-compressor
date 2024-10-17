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

from abc import abstractmethod

import habana_frameworks.torch.core as htcore
import torch
import torch.nn as nn

from .._core.scale_handler import create_scale_tensor
from .._quant_common.quant_config import ScaleFormat

descale_fcn = lambda x, scale: torch.mul(x, scale)
scale_fcn = lambda x, scale: torch.div(x, scale)
cast_fcn = lambda x, dtype: x.to(dtype=dtype)
cast_to_fp8_fcn = lambda x, dtype, scale_inv=None: torch.ops.hpu.cast_to_fp8_v2(x, scale_inv, False, False, dtype)[0]
cast_from_fp8_fcn = lambda x, dtype, scale=None: torch.ops.hpu.cast_from_fp8(x, scale, dtype)


class QuantDequantBase(nn.Module):
    def __init__(self, lp_dtype, hp_dtype="", *args, **kwargs):
        super(QuantDequantBase, self).__init__()
        self.lp_dtype = lp_dtype
        self.hp_dtype = hp_dtype
        self.scale_format = ScaleFormat.CONST
        if "scale_format" in kwargs:
            self.scale_format = kwargs["scale_format"]

    @abstractmethod
    def forward(self, *args, **kwargs):
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
        self.scale_inv = create_scale_tensor(scale_inv, self.scale_format)

    def forward(self, x):
        return cast_to_fp8_fcn(x, self.lp_dtype, self.scale_inv)

    def extra_repr(self) -> str:
        repr = super(QuantInput, self).extra_repr()
        return f"{repr}, scale_inv dtype={self.scale_inv.dtype}"


class DequantOutput(QuantDequantBase):
    def __init__(self, scale, lp_dtype, hp_dtype, *args, **kwargs):
        super(DequantOutput, self).__init__(lp_dtype, hp_dtype, *args, **kwargs)
        self.scale = create_scale_tensor(scale, self.scale_format)

    def forward(self, x):
        return cast_from_fp8_fcn(x, self.hp_dtype, self.scale)

    def extra_repr(self) -> str:
        repr = super(DequantOutput, self).extra_repr()
        return f"{repr}, scale dtype={self.scale.dtype}"


class QuantDequant(QuantDequantBase):
    def __init__(self, scale_inv, lp_dtype, hp_dtype, *args, **kwargs):
        super(QuantDequant, self).__init__(lp_dtype, hp_dtype, *args, **kwargs)
        self.scale_inv = create_scale_tensor(scale_inv, self.scale_format)
        self.scale = create_scale_tensor(1 / scale_inv, self.scale_format)

    def forward(self, x, *args, **kwargs):
        y = cast_to_fp8_fcn(x, self.lp_dtype, self.scale_inv)
        # mark_step is needed so fuser won't remove 2 consecutive casts.
        # will be removed once SW-196431 is implemented
        htcore.mark_step()
        z = cast_from_fp8_fcn(y, self.hp_dtype, self.scale)
        htcore.mark_step()
        return z

    def extra_repr(self) -> str:
        repr = super(QuantDequant, self).extra_repr()
        return f"{repr}, Quantize, and then dequantize"
