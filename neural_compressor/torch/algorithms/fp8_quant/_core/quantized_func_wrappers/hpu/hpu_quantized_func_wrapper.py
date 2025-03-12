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

from abc import ABCMeta, abstractmethod
from enum import Enum, auto

import torch

from ..quantized_func_wrapper import QuantizedFuncWrapperBase, OP_TYPE, QuantizedFuncWrapperFactory
from neural_compressor.torch.algorithms.fp8_quant._quant_common.quant_config import ScaleFormat
try:  # backwards compatibility for 1.16
    from habana_frameworks.torch.hpex.kernels import fp8_fused_sdpa
except ImportError:
    pass


import torch
FP8_DTYPE = torch.float8_e4m3fn
# def is_denormalized_fp8_cpu(fp8_tensor: torch.Tensor) -> torch.Tensor:
#     # fp8_tensor_hpu >> 3: RuntimeError: "rshift_cpu" not implemented for 'Float' 
#     fp8_tensor = fp8_tensor.cpu().view(torch.uint8)
#     exponent = (fp8_tensor >> 3) & 0xF
#     fraction = fp8_tensor & 0x7
#     mask = (exponent == 0) & (fraction != 0)
#     return mask.to("hpu")

# 7/8×(2^−6)
MAX_FP8SUBNORMAL = 0.875 * (2 ** -6)
def is_denormalized_fp8(fp8_tensor: torch.Tensor) -> torch.Tensor:
    # fp8_tensor_float = fp8_tensor.to(torch.float)
    fp8_tensor_float = torch.ops.hpu.cast_from_fp8(fp8_tensor, None, torch.float)
    
    return (fp8_tensor_float != 0) & (fp8_tensor_float.abs() <= MAX_FP8SUBNORMAL)

# # 7/8×(2^−6)
# MAX_FP8SUBNORMAL = 0.875 * (2 ** -6)
# def is_denormalized_fp8(fp8_tensor: torch.Tensor) -> torch.Tensor:
#     fp8_tensor_float = fp8_tensor.to(torch.float)
#     return (fp8_tensor_float != 0) & (fp8_tensor_float.abs() <= MAX_FP8SUBNORMAL)


def apply_ftz(fp8_tensor: torch.Tensor):
    assert fp8_tensor.dtype == FP8_DTYPE, f"Expected torch.float8_e4m3fnuz, got {fp8_tensor.dtype}"
    denormal_mask = is_denormalized_fp8(fp8_tensor)
    fp8_tensor[denormal_mask] = 0.0
    return fp8_tensor
    

class QuantizedHpuFuncWrapperBase(QuantizedFuncWrapperBase, metaclass=ABCMeta):
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



class QuantizedHpuMatmul(QuantizedHpuFuncWrapperBase):
    def get_default_quantized_func(self):
        return torch.ops.hpu.fp8_gemm_v2

    # only specific arguments are defined, to avoid having all other arguments defined in each call in patched modules.
    def __call__(self, input, other, out=None, out_dtype=torch.bfloat16, scale_input_inv=None, scale_other_inv=None):
        input = apply_ftz(input)
        other = apply_ftz(other)
        
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


class QuantizedHpuConv(QuantizedHpuFuncWrapperBase):
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


class QuantizedHpuSoftmax(QuantizedHpuFuncWrapperBase):
    def get_default_quantized_func(self):
        return torch.ops.hpu.softmax_fp8

    def get_scalar_quantized_func(self):
        # softmax custom op has different scalar impl name
        return self.get_default_quantized_func().Scalar_scales


class QuantizedHpuFSDPA(QuantizedHpuFuncWrapperBase):
    def get_default_quantized_func(self):
        return fp8_fused_sdpa

    def get_scalar_quantized_func(self):
        # FSDPA isn't optimized for scalar flavor due to complexity of specific torch op api selection
        return self.get_default_quantized_func()


class QuantizedHpuDynamicMoe(QuantizedHpuFuncWrapperBase):
    def get_default_quantized_func(self):
        return torch.ops.hpu.mixture_of_experts.fp8

    def get_scalar_quantized_func(self):
        return torch.ops.hpu.mixture_of_experts.fp8_scalars


class QuantizedHPUCastToFP8(QuantizedHpuFuncWrapperBase):
    def get_default_quantized_func(self):
        return torch.ops.hpu.cast_to_fp8_v2

    def __call__(self, *args, **kwargs):
        return self._quantized_func_(*args, **kwargs)[0]

class QuantizedHPUCastFromFP8(QuantizedHpuFuncWrapperBase):

    def __init__(self, scale_format):
        super().__init__(scale_format)

    def get_default_quantized_func(self):
        return torch.ops.hpu.cast_from_fp8

class QuantizedHpuDynamicMoeFusedWeights(QuantizedHpuFuncWrapperBase):
    def get_default_quantized_func(self):
        return torch.ops.hpu.mixture_of_experts.fp8_fused_weights
    def get_scalar_quantized_func(self):
        return torch.ops.hpu.mixture_of_experts.fp8_fused_weights_scalars


_OP_TYPE_HPU_QUANTIZED_WRAPPER_CLASSES = {OP_TYPE.LINEAR_GEMM : QuantizedHpuMatmul,
                                          OP_TYPE.MATMUL_GEMM: QuantizedHpuMatmul,
                                          OP_TYPE.SOFTMAX : QuantizedHpuSoftmax,
                                          OP_TYPE.CONV  : QuantizedHpuConv,
                                          OP_TYPE.FSDPA : QuantizedHpuFSDPA,
                                          OP_TYPE.CAST_TO_FP8 : QuantizedHPUCastToFP8,
                                          OP_TYPE.CAST_FROM_FP8 : QuantizedHPUCastFromFP8,
                                          OP_TYPE.DYNAMIC_MOE: QuantizedHpuDynamicMoe,
                                          OP_TYPE.DYNAMIC_MOE_FUSED_WEIGHTS: QuantizedHpuDynamicMoeFusedWeights,
                                          }

def init_hpu_quantized_func_wrapper_factory():
    QuantizedFuncWrapperFactory.initialize(_OP_TYPE_HPU_QUANTIZED_WRAPPER_CLASSES)
