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

from typing import Dict, Optional, Tuple, Any
from neural_compressor.torch.utils.auto_accelerator import auto_detect_accelerator

__all__ = [
    "IMOD_DICT",
    "ModuleInfo",
    "ModuleConfig",
    "ModuleType",
    "ModuleExtraConfig",
    "OBSERVER_PARAMS",
    "OBSERVER_TYPES",
    "PATCHED_MODULE_TABLE",
    "PATCHED_MODULE_TYPES_TABLE",
    "SCALING_METHODS_TABLE",
    "get_patched_module_table",
    "get_patched_module_type_table",
    "import_external_scaling_methods",
    "clear_external_scaling_methods",
]

# ==-------------------------------------------------------------------------==
# Patched Module Configurations
# ==-------------------------------------------------------------------------==
class ModuleInfo:
    """
    Configures a relation between a ModuleType key (from `_mod_types` dict in `common.py`)
    to a PatchedModule class.
    """
    def __init__(self, type, patched_module, should_measure_and_quant=True, *, supports_dynamic_quantization=False):
        self.type = type
        self.patched_module = patched_module
        self.should_measure_and_quant = should_measure_and_quant
        self.supports_dynamic_quantization = supports_dynamic_quantization

    def __repr__(self):
        return (
            f"ModuleInfo(type={self.type}, "
            f"patched_module={self.patched_module.__name__}), "
            f"should_measure_and_quant={self.should_measure_and_quant}, "
            f"supports_dynamic_quantization={self.supports_dynamic_quantization}"
        )


class ModuleConfig:
    """
    Used to encapsulate measure/quant data and consumed by ModuleExtraConfig(s)
    2 use cases:
    1. Hold the calculated scales per op-tensor
    2. Holds the quant/dequant ops per op-tensors
    """
    def __init__(self, inputs=(None,), outputs=(None,), params=None):
        self.inputs = inputs
        self.outputs = outputs
        self.params = params if params is not None else {}

    def __repr__(self):
        return f"(inputs={self.inputs}, outputs={self.outputs}, params={self.params})"


class ModuleExtraConfig:
    """
    Used to encapsulate measure/quant data and consumed by PatchedModule(s)
    2 use cases:
    1. Holds the observers per op-tensors (MEASURE mode)
    2. Holds the scale values and quant/dequant ops per op-tensor (QUANT mode)
    """
    def __init__(self, inputs=(None,), outputs=(None,), params=None, scale=None, config_params=None):
        self.inputs = inputs
        self.outputs = outputs
        self.params = params if params is not None else {}
        self.scale = scale
        self.config_params = config_params if config_params is not None else {}

    def __repr__(self):
        return f"(inputs={self.inputs}, outputs={self.outputs}, params={self.params}, scale={self.scale}, config_params={self.config_params})"


class ModuleType:
    """
    Describes the module's tensor related structure for quantization.

    self.num_inputs - Number inputs to be quantizeds (e.g. input of Linear).
    self.param_names - Names of module's const data tensors (e.g. weights of Linear). <Usage explanation>
    self.num_outputs - Number outputs to be quantizeds (e.g. output of Linear).
    self.required_output - Whether to measure the outputs or not.
    """
    def __init__(self, num_inputs, param_names, num_outputs, required_output):
        self.num_inputs = num_inputs
        self.param_names = param_names
        self.num_outputs = num_outputs
        self.required_output = required_output

    def __repr__(self):
        return (
            f"ModuleType(num_inputs={self.num_inputs}, param_names={self.param_names},"
            f"num_outputs={self.num_outputs}, required_output={self.required_output})"
        )


# ==-------------------------------------------------------------------------==
# Patched Module Tables
# ==-------------------------------------------------------------------------==

DEVICE_TYPES = ["hpu", "xpu", "cuda", "cpu"]
# Note: `PATCHED_MODULE_TABLE` is a nested dictionary with the following structure:
#   {device_type: {module_type: ModuleInfo}}
PATCHED_MODULE_TABLE: Dict[str, Dict[str, ModuleInfo]] = {key: {} for key in DEVICE_TYPES}
# Note: `PATCHED_MODULE_TYPES_TABLE` is a nested dictionary with the following structure:
#   {device_type: {module_type: ModuleType}}
PATCHED_MODULE_TYPES_TABLE: Dict[str, Dict[str, ModuleType]] = {key: {} for key in DEVICE_TYPES}


def get_patched_module_table(device_type: Optional[str] = None) -> Dict[str, ModuleInfo]:
    if device_type is None:
        device_type = auto_detect_accelerator().name()
    return PATCHED_MODULE_TABLE[device_type]


def get_patched_module_type_table(device_type: Optional[str] = None) -> Dict[str, ModuleType]:
    if device_type is None:
        device_type = auto_detect_accelerator().name()
    return PATCHED_MODULE_TYPES_TABLE[device_type]


# ==-------------------------------------------------------------------------==
# Scaling Method Tables
# ==-------------------------------------------------------------------------==

SCALING_METHODS_TABLE: Dict[str, Tuple[str, Any]] = {}


def import_external_scaling_methods():
    """Import the scaling methods out of tree.

    This function is called by the `register_scaling_methods` decorator and import
    the scaling methods registered out-of-tree to the `scaling_methods` dictionary.
    """
    from neural_compressor.torch.algorithms.fp8_quant._core.scale import (
        scaling_methods,
    )

    for scaling_method_name, (patch_module_type, cls) in SCALING_METHODS_TABLE.items():
        if scaling_method_name not in scaling_methods:
            scaling_methods[scaling_method_name] = {}
        scaling_methods[scaling_method_name][patch_module_type] = cls


def clear_external_scaling_methods():
    SCALING_METHODS_TABLE.clear()


# ==-------------------------------------------------------------------------==
# Measure Tables
# ==-------------------------------------------------------------------------==

OBSERVER_TYPES = {}
OBSERVER_PARAMS = {}
IMOD_DICT = {}