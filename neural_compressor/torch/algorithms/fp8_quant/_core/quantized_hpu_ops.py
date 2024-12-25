
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
    GEMM = auto(),
    SOFTMAX = auto()
    CONV = auto()
    FSDPA = auto()


class QuantizedHpuFuncWrapper(ABC):
    """
    Base class for wrapping calls to hpu custom fp8 ops.
    The concrete class object is created in patched module init in call to get_hpu_quantized_func_wrapper.
    Concrete class should define get_default_quantized_func method.
    Concrete class may override base class methods in case custom op logic is unique, see examples in concrete
    classes below.
    """
    def __init__(self, scale_format):
        self.set_quantized_func(scale_format)
        self.quantized_func_args = None

    @abstractmethod
    def get_default_quantized_func(self):
        raise NotImplementedError()

    def get_scalar_quantized_func(self):
        return self.get_default_quantized_func().scalar

    def set_quantized_func(self, scale_format):
        if scale_format == ScaleFormat.SCALAR:
                self._quantized_func_ = self.get_scalar_quantized_func()
        elif scale_format == ScaleFormat.CONST:
            self._quantized_func_ = self.get_default_quantized_func()
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

    def __init__(self, scale_format):
        # FSDPA isn't optimized for scalar flavor due to complexity of specific torch op api selection
        self._quantized_func_ = self.get_default_quantized_func()

    def get_default_quantized_func(self):
        return fp8_fused_sdpa

    def get_scalar_quantized_func(self):
        raise NotImplementedError()

_OP_TYPE_HPU_QUANTIZED_WRAPPER_CLASSES = {OP_TYPE.GEMM : QuantizedHpuMatmul,
                                          OP_TYPE.SOFTMAX : QuantizedHpuSoftmax,
                                          OP_TYPE.CONV  : QuantizedHpuConv,
                                          OP_TYPE.FSDPA : QuantizedHpuFSDPA
                                          }

def get_hpu_quantized_func_wrapper(op_type, scale_format):
    quantized_hpu_wrapper_class = _OP_TYPE_HPU_QUANTIZED_WRAPPER_CLASSES[op_type]
    return quantized_hpu_wrapper_class(scale_format)