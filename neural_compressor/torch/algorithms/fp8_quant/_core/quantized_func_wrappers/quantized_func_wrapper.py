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

from abc import ABC, abstractmethod
from enum import Enum, auto


class OP_TYPE(Enum):
    # class per custom fp8 ops used in patched modules logic
    LINEAR_GEMM = auto()
    MATMUL_GEMM = auto()
    SOFTMAX = auto()
    BLOCK_SOFTMAX_CONST_MAX = auto()
    CONV = auto()
    FSDPA = auto()
    CAST_TO_FP8 = auto()
    CAST_FROM_FP8 = auto()
    DYNAMIC_MOE = auto()
    DYNAMIC_MOE_FUSED_WEIGHTS = auto()
    QUANT = auto()
    DEQUANT = auto()
    QUANT_PC = auto()
    DEQUANT_PC = auto()


class QuantizedFuncWrapperBase(ABC):
    """
    Base class for wrapping calls to custom fp8 ops.
    The concrete class object is created in patched module init in call to get_<device>_quantized_func_wrapper.
    Concrete class should define get_default_quantized_func method.
    Concrete class may override base class methods in case custom op logic is unique, see examples in concrete
    classes implementation.
    """
    @abstractmethod
    def get_default_quantized_func(self):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self._quantized_func_(*args, **kwargs)


class QuantizedFuncWrapperFactory():
    """
    A Factory object to create func wrappers objects.
    This is a singleton and it creates single object per quantized func wrapper class.
    This is done to avoid unnecessary duplication of quantized func wrapper objects since, since they are all identical.
    """

    __is_initialized = False

    # using a global map and the below get_quantized_func_wrapper method,
    # ensures only a single object of each quantized func wrapper concrete class will be created.
    __quantized_func_wrapper_instances = {}

    __device_func_wrappers_mapping = {}

    def __new__(cls):
        raise TypeError("The {} class can't be instantiated directly".format(cls.__name__))

    @classmethod
    def initialize(cls, device_quantized_func_wrapper_dict):
        if not cls.__is_initialized:
            if not device_quantized_func_wrapper_dict:
                raise ValueError("Expected non empty device_quantized_func_wrapper_dict")
            cls.__device_func_wrappers_mapping = dict(device_quantized_func_wrapper_dict)
            cls.__is_initialized = True


    @classmethod
    def get_quantized_func_wrapper_object(cls, op_type, scale_format, is_dynamic=False):
        if op_type not in cls.__quantized_func_wrapper_instances:
            quantized_wrapper_class = cls.__device_func_wrappers_mapping[op_type]
            cls.__quantized_func_wrapper_instances[op_type] = quantized_wrapper_class(scale_format, is_dynamic)

        return cls.__quantized_func_wrapper_instances[op_type]

    @classmethod
    def clear(cls):
        cls.__quantized_func_wrapper_instances.clear()
        cls.__is_initialized = True
