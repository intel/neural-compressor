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

from enum import Enum, auto

import torch

from neural_compressor.torch.utils.auto_accelerator import auto_detect_accelerator
from .quantized_func_wrapper import QuantizedFuncWrapperFactory


"""
Functions to interact with QuantizedFuncWrapperFactory singleton object
"""

def get_quantized_func_wrapper(op_type, scale_format, is_dynamic=False):
    return QuantizedFuncWrapperFactory.get_quantized_func_wrapper_object(op_type, scale_format, is_dynamic)


def init_quantized_func_wrapper_factory():
    curr_accelerator = auto_detect_accelerator()
    device_name = curr_accelerator.name()
    if device_name == "hpu":
        from .hpu.hpu_quantized_func_wrapper import init_hpu_quantized_func_wrapper_factory
        init_hpu_quantized_func_wrapper_factory()
    elif device_name == "xpu":
        from .xpu.xpu_quantized_func_wrapper import init_xpu_quantized_func_wrapper_factory
        init_xpu_quantized_func_wrapper_factory()
    elif device_name == "cpu":
        from .cpu.cpu_quantized_func_wrapper import init_cpu_quantized_func_wrapper_factory
        init_cpu_quantized_func_wrapper_factory()
    else:
        raise ValueError("Unknown device type - {}".format(device_name))


def clear_quantized_func_wrapper_factory():
    QuantizedFuncWrapperFactory.clear()
