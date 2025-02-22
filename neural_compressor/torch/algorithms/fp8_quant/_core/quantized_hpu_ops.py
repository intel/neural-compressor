
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

from .._quant_common.quant_config import ScaleFormat
from ..utils.logger import logger

try:  # backwards compatibility for 1.16
    from habana_frameworks.torch.hpex.kernels import fp8_fused_sdpa
except ImportError:
    pass

import torch

from abc import ABC, abstractmethod
from enum import Enum, auto

class OP_TYPE(Enum):
    # class per hpu custom fp8 ops used in patched modules logic
    GEMM = auto()
    SOFTMAX = auto()
    CONV = auto()
    FSDPA = auto()
    DYNAMIC_MOE = auto()
    DYNAMIC_MOE_FUSED_WEIGHTS = auto()


class QuantizedHpuFuncWrapper(ABC):
    """
    Base class for wrapping calls to hpu custom fp8 ops.
    The concrete class object is created in patched module init in call to get_hpu_quantized_func_wrapper.
    Concrete class should define get_default_quantized_func method.
    Concrete class may override base class methods in case custom op logic is unique, see examples in concrete
    classes below.
    """
    def __init__(self, scale_format):
        self._quantized_func_ = self.get_quantized_func(scale_format)

    @abstractmethod
    def get_default_quantized_func(self):
        raise NotImplementedError()

    def get_scalar_quantized_func(self):
        return self.get_default_quantized_func().scalar

    def get_quantized_func(self, scale_format):
        if scale_format == ScaleFormat.SCALAR:
                return self.get_scalar_quantized_func()
        elif scale_format == ScaleFormat.CONST:
            return self.get_default_quantized_func()
        else:
            raise ValueError("Unexpected scale format - {}".format(scale_format))

    def __call__(self, *args, **kwargs):
        return self._quantized_func_(*args, **kwargs)


class QuantizedHpuMatmul(QuantizedHpuFuncWrapper):
    def get_default_quantized_func(self):
        return torch.ops.hpu.fp8_gemm_v2

    # only specific arguments are defined, to avoid having all other arguments defined in each call in patched modules.
    def __call__(self, input, other, out=None, out_dtype=torch.bfloat16, scale_input_inv=None, scale_other_inv=None):
        return self._quantized_func_(input,
                                     False,
                                     other,
                                     False,
                                     out,
                                     out_dtype,
                                     scale_input_inv,
                                     scale_other_inv,
                                     None,
                                     False)


class QuantizedHpuConv(QuantizedHpuFuncWrapper):
    def get_default_quantized_func(self):
        return torch.ops.hpu.conv2d_fp8

    @staticmethod
    def to_list_if_necessary(param):
        return param if hasattr(param, "__iter__") else [param] * 2

    # only specific arguments are defined, to avoid having all other arguments defined in each call in patched modules.
    def __call__(self,
                 input,
                 weight,
                 bias,
                 stride,
                 padding,
                 dilation,
                 groups,
                 out_dtype=torch.bfloat16,
                 scale_input_inv=None,
                 scale_other_inv=None):

        return self._quantized_func_(input=input,
                                     weight=weight,
                                     bias=bias,
                                     stride=self.to_list_if_necessary(stride),
                                     padding=self.to_list_if_necessary(padding),
                                     dilation=self.to_list_if_necessary(dilation),
                                     groups=groups,
                                     out_dtype=out_dtype,
                                     scale_input=scale_input_inv,
                                     scale_weight=scale_other_inv)


class QuantizedHpuSoftmax(QuantizedHpuFuncWrapper):
    def get_default_quantized_func(self):
        return torch.ops.hpu.softmax_fp8

    def get_scalar_quantized_func(self):
        # softmax custom op has different scalar impl name
        return self.get_default_quantized_func().Scalar_scales


class QuantizedHpuFSDPA(QuantizedHpuFuncWrapper):
    def get_default_quantized_func(self):
        return fp8_fused_sdpa

    def get_scalar_quantized_func(self):
        # FSDPA isn't optimized for scalar flavor due to complexity of specific torch op api selection
        return self.get_default_quantized_func()


class QuantizedHpuDynamicMoe(QuantizedHpuFuncWrapper):
    def get_default_quantized_func(self):
        return torch.ops.hpu.mixture_of_experts.fp8

    def get_scalar_quantized_func(self):
        return torch.ops.hpu.mixture_of_experts.fp8_scalars


class QuantizedHpuDynamicMoeFusedWeights(QuantizedHpuFuncWrapper):
    def get_default_quantized_func(self):
        return torch.ops.hpu.mixture_of_experts.fp8_fused_weights

    def get_scalar_quantized_func(self):
        return torch.ops.hpu.mixture_of_experts.fp8_fused_weights_scalars


_OP_TYPE_HPU_QUANTIZED_WRAPPER_CLASSES = {OP_TYPE.GEMM : QuantizedHpuMatmul,
                                          OP_TYPE.SOFTMAX : QuantizedHpuSoftmax,
                                          OP_TYPE.CONV  : QuantizedHpuConv,
                                          OP_TYPE.FSDPA : QuantizedHpuFSDPA,
                                          OP_TYPE.DYNAMIC_MOE: QuantizedHpuDynamicMoe,
                                          OP_TYPE.DYNAMIC_MOE_FUSED_WEIGHTS: QuantizedHpuDynamicMoeFusedWeights,
                                          }

class QuantizedFuncWrapperFactory():
    """
    A Factory object to create func wrappers objects.
    This is a singleton and it creates single object per quantized func wrapper class.
    This is done to avoid unnecessary duplication of quantized func wrapper objects since, since they are all identical.
    """

    _factory_instance = None

    # using a global map and the below get_quantized_func_wrapper method,
    # ensures only a single object of each quantized func wrapper concrete class will be created.
    _quantized_func_wrapper_instances = {}

    def __new__(cls):
        if cls._factory_instance is None:
            cls._factory_instance = super().__new__(cls)
        return cls._factory_instance

    def get_quantized_func_wrapper(self, op_type, scale_format):
        if op_type not in self._quantized_func_wrapper_instances:
            quantized_hpu_wrapper_class = _OP_TYPE_HPU_QUANTIZED_WRAPPER_CLASSES[op_type]
            self._quantized_func_wrapper_instances[op_type] = quantized_hpu_wrapper_class(scale_format)

        return self._quantized_func_wrapper_instances[op_type]

    def clear(self):
        self._quantized_func_wrapper_instances.clear()

    def __del__(self):
        self._factory_instance = None



def get_hpu_quantized_func_wrapper(op_type, scale_format):
    return QuantizedFuncWrapperFactory().get_quantized_func_wrapper(op_type, scale_format)