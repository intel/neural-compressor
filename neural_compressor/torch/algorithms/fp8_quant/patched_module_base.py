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

"""Base class for patched modules and helper functions for registering patched modules."""

from typing import Union, List, Type, Optional
from abc import abstractmethod
import torch
from neural_compressor.torch.algorithms.fp8_quant._core.common import (
    QuantMode,
)
from neural_compressor.common import utils as inc_utils
from neural_compressor.torch.algorithms.fp8_quant.utils import (
    helper_mod_register,
)
from neural_compressor.torch.algorithms.fp8_quant._quant_common.helper_modules import (
    set_attrs_from_orig_model,
)

from neural_compressor.torch.algorithms.fp8_quant.model_configs import (
    ModuleInfo,
    ModuleType,
    ModuleExtraConfig,
    PATCHED_MODULE_TABLE,
    PATCHED_MODULE_TYPES_TABLE,
    DEVICE_TYPES,
)

__all__ = [
    "PatchedModuleBase",
    "register_patched_module",
]


def register_patched_module(
    supported_float_module_types: Union[str, Type[torch.nn.Module], List[str], List[Type[torch.nn.Module]]],
    device_types: Optional[Union[str, List[str]]] = "all",
) -> None:
    """Register a patched module with support for specific float module types and device types.

    Args:
        supported_float_module_types (Union[str, Type[torch.nn.Module], List[Union[str, Type[torch.nn.Module]]]]):
            The float module types to support. Can be a single string, a single torch.nn.Module type,
            or a list containing strings and/or torch.nn.Module types.
        device_types (Optional[Union[str, List[str]]], optional):
            The device types to support. Can be a single string or a list of strings.
            Defaults to "all", which means the patched module will be registered for *all device* types.

    Examples:
        1. Register a patched module for `ModuleFusedSDPA` on HPU device:
        @register_patched_module("ModuleFusedSDPA", device_types="hpu")
        class PatchedModuleFusedSDPA(PatchedModuleBase):
            ...

        2. Register a patched module for `Linear` and `RowParallelLinear` on all devices:
        @register_patched_module(["Linear", "RowParallelLinear"])
        class PatchedLinear(PatchedModuleBase):
            ...

    """

    def decorator(patch_mdoule_cls: "PatchedModuleBase") -> "PatchedModuleBase":
        nonlocal supported_float_module_types, device_types
        if not isinstance(supported_float_module_types, list):
            supported_float_module_types = [supported_float_module_types]
        if not isinstance(device_types, list):
            device_types = [device_types]
            if "all" in device_types:
                device_types = DEVICE_TYPES
        for device_type in device_types:
            for supported_type in supported_float_module_types:
                supported_type_name = (
                    supported_type is isinstance(supported_type, str) and supported_type or supported_type.__name__
                )
                module_info = patch_mdoule_cls.get_module_info()
                module_type = patch_mdoule_cls.get_module_type()
                PATCHED_MODULE_TABLE[device_type][supported_type_name] = module_info
                PATCHED_MODULE_TYPES_TABLE[device_type][supported_type_name] = module_type
                inc_utils.logger.info(
                    (
                        f"For type {supported_type}, registered patched module with \n"
                        f"    module_info: {module_info}\n"
                        f"    model_type: {module_type}\n"
                        f"    device_type: {device_type}"
                    )
                )
                # Add the new module to the `helper_mods` dict
                _create_and_register_helper_module_class(supported_type.__name__)
        return patch_mdoule_cls

    return decorator


class PatchedModuleBase(torch.nn.Module):
    """Base class for patched modules.

    The patched module is used to replace the original float module in the model at `prepare` and `convert` stages.
    To make the patched module compatible with the module-agnostic patching process, the patched module should
    provide a set of module configurations by implementing the following abstract methods:
        - get_type
        - get_module_type
    The patched module should also implement the following methods for its specific functionalities:
        - forward_measure
        - forward_quant
        - forward_qdq(Optional), only required when using Q-DQ mode for quantization stage.
    """

    def __init__(
        self,
        mod: torch.nn.Module,
        mod_extra_config: ModuleExtraConfig,
        name: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the patched module."""
        super().__init__()
        self.name = name
        set_attrs_from_orig_model(self, mod, mod_extra_config)
        self.mod_extra_config = mod_extra_config
        if self.quantization_mode in (QuantMode.MEASURE, QuantMode.SHAPE):
            self.forward = self.forward_measure
        elif self.quantization_mode == QuantMode.QUANTIZE:
            if self.fake_quant:
                self.forward = self.forward_qdq
            else:
                self.forward = self.forward_quant

    def forward_qdq(self, *args, **kwargs):
        """Forward function with Q-DQ mode for quantization stage."""

    @abstractmethod
    def forward_measure(self, *args, **kwargs):
        """Forward function for measurement stage."""
        raise NotImplementedError("forward_measure is not implemented")

    @abstractmethod
    def forward_quant(self, *args, **kwargs):
        """Forward function for quantization stage."""
        raise NotImplementedError("forward_quant is not implemented")

    @classmethod
    def get_module_info(cls) -> ModuleInfo:
        """Return the module info for the module.

        For example, for linear module, the module info is: ModuleInfo(type="linear", patched_module=cls).
        """
        return ModuleInfo(type=cls.get_type(), patched_module=cls)

    @classmethod
    @abstractmethod
    def get_type(cls) -> str:
        """Return the type of the patched module.

        Multiple patched modules can have the same type, and share the same scling methods.
        """
        raise NotImplementedError("`get_type` is not implemented")

    @classmethod
    @abstractmethod
    def get_module_type(cls) -> ModuleType:
        """Return the module type for the module.

        The module type is used to determine the number of inputs, outputs, and parameters of the module.
        For example, for linear module, the module type is: ModuleType(1, ["weight"], 1, False).
        """
        raise NotImplementedError("`get_module_type` is not implemented")

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"    quantization_mode={self.quantization_mode}, \n"
            f"    module_info={self.get_module_info()}, \n"
            f"    module_type={self.get_module_type()})\n"
        )


def _create_and_register_helper_module_class(name):
    @helper_mod_register(name=name)
    class _GenericModule(torch.nn.Module):
        def __init__(self, patched_mod, *args, **kwargs):
            super().__init__()
            self.__dict__.update(patched_mod.__dict__)
            self.extra_repr = patched_mod.extra_repr_org

    _GenericModule.__name__ = name
    return _GenericModule
