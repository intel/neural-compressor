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

import torch
from torchao.quantization.quant_primitives import (
    _quantize_affine_float8,
    _dequantize_affine_float8,
)

from abc import ABCMeta



class QuantizedCPUFuncWrapperBase(QuantizedFuncWrapperBase, metaclass=ABCMeta):
    """
    Placeholder for base class for CPU quantized func wrapper.
    """
    def __init__(self, scale_format, is_dynamic=False):
        self._quantized_func_ = self.get_default_quantized_func()


class QuantizedCPUQuant(QuantizedCPUFuncWrapperBase):

    def get_default_quantized_func(self):
        return _quantize_affine_float8

    def __call__(self, input, scale, zero_point=None, axis=0, quant_min=None, quant_max=None, dtype=torch.float8_e4m3fn):
        return self._quantized_func_(tensor=input, scale=torch.tensor(scale), float8_dtype=dtype)


class QuantizedCPUQuantPC(QuantizedCPUFuncWrapperBase):

    def get_default_quantized_func(self):
        return _quantize_affine_float8

    def __call__(self, input, scale, zero_point=None, axis=0, quant_min=None, quant_max=None, dtype=torch.float8_e4m3fn):
        return self._quantized_func_(tensor=input, scale=scale.view((-1, 1)), float8_dtype=dtype)


class QuantizedCPUDeQuant(QuantizedCPUFuncWrapperBase):

    def get_default_quantized_func(self):
        return _dequantize_affine_float8

    def __call__(self, input, scale, zero_point=None, axis=0, quant_min=None, quant_max=None, dtype=torch.float8_e4m3fn, out_dtype=torch.bfloat16):
        return self._quantized_func_(tensor=input, scale=torch.tensor(scale), output_dtype=out_dtype)


class QuantizedCPUDeQuantPC(QuantizedCPUFuncWrapperBase):

    def get_default_quantized_func(self):
        return _dequantize_affine_float8

    def __call__(self, input, scale, zero_point=None, axis=0, quant_min=None, quant_max=None, dtype=torch.float8_e4m3fn, out_dtype=torch.bfloat16):
        return self._quantized_func_(tensor=input, scale=scale.view((1, -1)), output_dtype=out_dtype)


_OP_TYPE_CPU_QUANTIZED_WRAPPER_CLASSES = {
                                          OP_TYPE.QUANT: QuantizedCPUQuant,
                                          OP_TYPE.DEQUANT: QuantizedCPUDeQuant,
                                          OP_TYPE.QUANT_PC: QuantizedCPUQuantPC,
                                          OP_TYPE.DEQUANT_PC: QuantizedCPUDeQuantPC,
                                         }


def init_cpu_quantized_func_wrapper_factory():
    QuantizedFuncWrapperFactory.initialize(_OP_TYPE_CPU_QUANTIZED_WRAPPER_CLASSES)
