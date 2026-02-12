# Copyright (c) 2025 Intel Corporation
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

from ..quantized_func_wrapper import QuantizedFuncWrapperBase, OP_TYPE, QuantizedFuncWrapperFactory
from neural_compressor.torch.algorithms.fp8_quant._quant_common.quant_config import ScaleFormat

import torch

from abc import ABCMeta



class QuantizedXPUFuncWrapperBase(QuantizedFuncWrapperBase, metaclass=ABCMeta):
    """
    Placeholder for base class for XPU quantized func wrapper.
    """
    def __init__(self, scale_format, is_dynamic=False):
        self._quantized_func_ = self.get_default_quantized_func()

class QuantizedXPUMatmul(QuantizedXPUFuncWrapperBase):
    def get_default_quantized_func(self):
        # TODO FSW-11669 modify func once fp8_gemm_v2 is implemented
        return torch.ops.torch_ipex.fp8_gemm

    # only specific arguments are defined, to avoid having all other arguments defined in each call in patched modules.
    def __call__(self, input, other, out=None, out_dtype=torch.bfloat16, scale_input_inv=None, scale_other_inv=None):
        # TODO FSW-11669 modify call arguments once fp8_gemm_v2 is implemented
        # Current ipex ops fp8_gemm API is misaligned to hpu ops API.
        # below args are according to ipex ops to allow basic unit testing, but won't support integration in
        # INC patched modules.
        return self._quantized_func_(input,
                                     torch.bfloat16,
                                     0,
                                     other,
                                     torch.bfloat16,
                                     1,
                                     None, # bias
                                     scale_input_inv,
                                     scale_other_inv
                                    )


class QuantizedXPUCastToFP8Base(QuantizedXPUFuncWrapperBase):
    def get_default_quantized_func(self):
        return torch.ops.torch_ipex.cast_to_fp8

    def __call__(self, *args, **kwargs):
        return self._quantized_func_(*args, **kwargs)[0]

class QuantizedXPUCastFromFP8Base(QuantizedXPUFuncWrapperBase):
    def get_default_quantized_func(self):
        return torch.ops.torch_ipex.cast_from_fp8


class QuantizedXPUQuant(QuantizedXPUFuncWrapperBase):

    def get_default_quantized_func(self):
        return torch.ops.quantized_decomposed.quantize_per_tensor

    def get_scalar_quantized_func(self):
        return self.get_default_quantized_func()

    def __call__(self, input, scale, zero_point=None, axis=0, quant_min=None, quant_max=None, dtype=torch.float8_e4m3fn):
        return self._quantized_func_(input, scale, zero_point, quant_min, quant_max, dtype=dtype)


class QuantizedXPUDeQuant(QuantizedXPUFuncWrapperBase):

    def get_default_quantized_func(self):
        return torch.ops.quantized_decomposed.dequantize_per_tensor

    def get_scalar_quantized_func(self):
        return self.get_default_quantized_func()

    def __call__(self, input, scale, zero_point=None, axis=0, quant_min=None, quant_max=None, dtype=torch.float8_e4m3fn, out_dtype=torch.bfloat16):
        return self._quantized_func_(input, scale, zero_point, quant_min, quant_max, dtype=dtype, out_dtype=out_dtype)


class QuantizedXPUQuantPC(QuantizedXPUFuncWrapperBase):

    def get_default_quantized_func(self):
        return torch.ops.quantized_decomposed.quantize_per_channel

    def get_scalar_quantized_func(self):
        return self.get_default_quantized_func()

    def __call__(self, input, scale, zero_point=None, axis=0, quant_min=None, quant_max=None, dtype=torch.float8_e4m3fn):
        return self._quantized_func_(input, scale, zero_point, axis, quant_min, quant_max, dtype=dtype)


class QuantizedXPUDeQuantPC(QuantizedXPUFuncWrapperBase):

    def get_default_quantized_func(self):
        return torch.ops.quantized_decomposed.dequantize_per_channel

    def get_scalar_quantized_func(self):
        return self.get_default_quantized_func()

    def __call__(self, input, scale, zero_point=None, axis=0, quant_min=None, quant_max=None, dtype=torch.float8_e4m3fn, out_dtype=torch.bfloat16):
        return self._quantized_func_(input, scale, zero_point, axis, quant_min, quant_max, dtype=dtype, out_dtype=out_dtype)

_OP_TYPE_XPU_QUANTIZED_WRAPPER_CLASSES = {
                                          OP_TYPE.LINEAR_GEMM : QuantizedXPUMatmul,
                                          OP_TYPE.MATMUL_GEMM : QuantizedXPUMatmul,
                                          OP_TYPE.CAST_TO_FP8 : QuantizedXPUCastToFP8Base,
                                          OP_TYPE.CAST_FROM_FP8 : QuantizedXPUCastFromFP8Base,
                                          OP_TYPE.QUANT: QuantizedXPUQuant,
                                          OP_TYPE.DEQUANT: QuantizedXPUDeQuant,
                                          OP_TYPE.QUANT_PC: QuantizedXPUQuantPC,
                                          OP_TYPE.DEQUANT_PC: QuantizedXPUDeQuantPC,
                                         }


def init_xpu_quantized_func_wrapper_factory():
    QuantizedFuncWrapperFactory.initialize(_OP_TYPE_XPU_QUANTIZED_WRAPPER_CLASSES)
