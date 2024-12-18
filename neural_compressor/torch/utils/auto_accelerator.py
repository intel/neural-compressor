# Copyright (c) 2023-2024 Microsoft Corporation and Intel Corporation

# This code is based on Microsoft Corporation's DeepSpeed library and
# the accelerators implementation in this library. It has been modified
# from its original forms to simplify and adapt it for use in
# the Intel® Neural Compressor.

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

# NOTICE: The design adapted from:
# https://github.com/microsoft/DeepSpeed/blob/master/accelerator/abstract_accelerator.py.
"""Auto Accelerator Module."""


# To keep it simply, only add the APIs we need.

import os
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Callable, List

import torch

from neural_compressor.common.utils import LazyImport, logger

htcore = LazyImport("habana_frameworks.torch.core")

PRIORITY_HPU = 100
PRIORITY_XPU = 95
PRIORITY_CUDA = 90
PRIORITY_CPU = 80


class AcceleratorRegistry:
    """Accelerator Registry."""

    registered_accelerators = {}

    @classmethod
    def register_accelerator_impl(cls, name: str, priority: float = 0):
        """Register new accelerator implementation.

        Usage example:
            @AcceleratorRegistry.register_accelerator(name="cpu", priority=100)
            class CPU_Accelerator:
                ...

        Args:
            name: the accelerator name.
            priority: priority: the priority of the accelerator. A larger number indicates a higher priority,
        """

        def decorator(accelerator_cls):
            if accelerator_cls.is_available():
                cls.registered_accelerators.setdefault(name, {})
                cls.registered_accelerators[name] = (accelerator_cls, priority)
            return accelerator_cls

        return decorator

    @classmethod
    def get_sorted_accelerators(cls) -> List["Auto_Accelerator"]:
        """Get registered accelerators sorted by priority."""
        accelerator_pairs = cls.registered_accelerators.values()
        sorted_accelerators_pairs = sorted(accelerator_pairs, key=lambda x: x[1], reverse=True)
        sorted_accelerators = [pair[0] for pair in sorted_accelerators_pairs]
        return sorted_accelerators

    @classmethod
    def get_accelerator_cls_by_name(cls, name: str) -> "Auto_Accelerator":
        """Get accelerator by name."""
        accelerator_cls, _ = cls.registered_accelerators.get(name, (None, None))
        return accelerator_cls


accelerator_registry = AcceleratorRegistry()


def register_accelerator(name: str, priority: float = 0) -> Callable[..., Any]:
    """Register new accelerator.

    Usage example:
        @register_accelerator(name="cuda", priority=100)
        class CUDA_Accelerator:
            ...

    Args:
        name: the accelerator name.
        priority: the priority of the accelerator. A larger number indicates a higher priority,
    """
    return accelerator_registry.register_accelerator_impl(name=name, priority=priority)


class Auto_Accelerator(ABC):  # pragma: no cover
    """Auto Accelerator Base class."""

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Check if the accelerator is available."""
        pass

    @abstractmethod
    def name(self) -> str:
        """Get the accelerator name."""
        pass

    @abstractmethod
    def device_name(self, device_indx) -> str:
        """Get the device name."""
        pass

    @abstractmethod
    def set_device(self, device_index):
        """Set the device."""
        pass

    @abstractmethod
    def current_device(self):
        """Get the current device."""
        pass

    @abstractmethod
    def current_device_name(self):
        """Get the current device name."""
        pass

    @abstractmethod
    def device(self, device_index=None):
        """Get the device."""
        pass

    @abstractmethod
    def empty_cache(self):
        """Empty the cache."""
        pass

    @abstractmethod
    def synchronize(self):
        """Synchronize the accelerator."""
        pass


@register_accelerator(name="cpu", priority=PRIORITY_CPU)
class CPU_Accelerator(Auto_Accelerator):
    """CPU Accelerator."""

    def __init__(self) -> None:
        """Initialize CPU Accelerator."""
        self._name = "cpu"

    def name(self) -> str:
        """Get the accelerator name."""
        return self._name

    @classmethod
    def is_available(cls) -> bool:
        """Always return True."""
        return True

    def device_name(self, device_indx) -> str:
        """Get the device name."""
        return "cpu"

    def set_device(self, device_index):
        """Do nothing."""
        pass

    def current_device(self):
        """Get the current device."""
        return "cpu"

    def current_device_name(self):
        """Get the current device name."""
        return "cpu"

    def device(self, device_index=None):
        """Do nothing."""
        pass

    def empty_cache(self):
        """Do nothing."""
        pass

    def synchronize(self):
        """Do nothing."""
        pass


@register_accelerator(name="cuda", priority=PRIORITY_CUDA)
class CUDA_Accelerator(Auto_Accelerator):  # pragma: no cover
    """CUDA Accelerator."""

    def __init__(self) -> None:
        """Initialize CUDA Accelerator."""
        self._name = "cuda"

    def name(self) -> str:
        """Get the accelerator name."""
        return self._name

    @classmethod
    def is_available(cls) -> bool:
        """Check if the 'cuda' device is available."""
        return torch.cuda.is_available()

    def device_name(self, device_indx) -> str:
        """Returns the name of the 'cuda' device with the given index."""
        if device_indx is None:
            return "cuda"
        return f"cuda:{device_indx}"

    def synchronize(self):
        """Synchronizes the 'cuda' device."""
        return torch.cuda.synchronize()

    def set_device(self, device_index):
        """Sets the current 'cuda' device to the one with the given index."""
        return torch.cuda.set_device(device_index)

    def current_device(self):
        """Returns the index of the current 'cuda' device."""
        return torch.cuda.current_device()

    def current_device_name(self):
        """Returns the name of the current 'cuda' device."""
        return "cuda:{}".format(torch.cuda.current_device())

    def device(self, device_index=None):
        """Returns a torch.device object for the 'cuda' device with the given index."""
        return torch.cuda.device(device_index)

    def empty_cache(self):
        """Empties the cuda cache."""
        return torch.cuda.empty_cache()


@register_accelerator(name="xpu", priority=PRIORITY_XPU)
class XPU_Accelerator(Auto_Accelerator):  # pragma: no cover
    """XPU Accelerator."""

    def __init__(self) -> None:
        """Initialize XPU Accelerator."""
        self._name = "xpu"

    def name(self) -> str:
        """Get the accelerator name."""
        return self._name

    @classmethod
    def is_available(cls) -> bool:
        """Checks if the 'xpu' device is available.

        Returns:
            bool: True if the 'xpu' device is available, False otherwise.
        """
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return True
        else:
            return False

    def device_name(self, device_indx) -> str:
        """Returns the name of the 'xpu' device with the given index.

        Args:
            device_indx (int): The index of the 'xpu' device.

        Returns:
            str: The name of the 'xpu' device.
        """
        if device_indx is None:
            return "xpu"
        return f"xpu:{device_indx}"

    def synchronize(self):
        """Synchronizes the 'xpu' device."""
        return torch.xpu.synchronize()

    def set_device(self, device_index):
        """Sets the current 'xpu' device to the one with the given index.

        Args:
            device_index (int): The index of the 'xpu' device.
        """
        return torch.xpu.set_device(device_index)

    def current_device(self):
        """Returns the index of the current 'xpu' device.

        Returns:
            int: The index of the current 'xpu' device.
        """
        return torch.xpu.current_device()

    def current_device_name(self):
        """Returns the name of the current 'xpu' device.

        Returns:
            str: The name of the current 'xpu' device.
        """
        return "xpu:{}".format(torch.xpu.current_device())

    def device(self, device_index=None):
        """Returns a torch.device object for the 'xpu' device with the given index.

        Args:
            device_index (int, optional): The index of the 'xpu' device. Defaults to None.

        Returns:
            torch.device: The torch.device object for the 'xpu' device.
        """
        return torch.xpu.device(device_index)

    def empty_cache(self):
        """Empties the xpu cache."""
        return torch.xpu.empty_cache()


@register_accelerator(name="hpu", priority=PRIORITY_HPU)
class HPU_Accelerator(Auto_Accelerator):  # pragma: no cover
    """HPU Accelerator."""

    def __init__(self) -> None:
        """Initialize HPU Accelerator."""
        self._name = "hpu"

    def name(self) -> str:
        """Get the accelerator name."""
        return self._name

    @classmethod
    def is_available(cls) -> bool:
        """Checks if the 'hpu' device is available."""
        from .environ import is_hpex_available

        if is_hpex_available():
            return torch.hpu.is_available()
        else:
            return False

    def device_name(self, device_indx) -> str:
        """Returns the name of the 'hpu' device with the given index."""
        if device_indx is None:
            return "hpu"
        return f"hpu:{device_indx}"

    def synchronize(self):
        """Synchronizes the 'hpu' device."""
        logger.debug("Calling `htcore.mark_step()` and `torch.hpu.synchronize()`.")
        htcore.mark_step()
        return torch.hpu.synchronize()

    def set_device(self, device_index):
        """Sets the current 'hpu' device to the one with the given index."""
        try:
            torch.hpu.set_device(device_index)
        except Exception as e:
            logger.warning(e)

    def current_device(self):
        """Returns the index of the current 'hpu' device."""
        return torch.hpu.current_device()

    def current_device_name(self):
        """Returns the name of the current 'hpu' device."""
        return "hpu:{}".format(torch.hpu.current_device())

    def device(self, device_index=None):
        """Returns a torch.device object for the 'hpu' device with the given index."""
        return torch.hpu.device(device_index)

    def empty_cache(self):
        """Empties the hpu cache."""
        try:
            torch.hpu.empty_cache()
        except Exception as e:
            logger.warning(e)


@lru_cache()
def auto_detect_accelerator(device_name="auto") -> Auto_Accelerator:
    """Automatically detects and selects the appropriate accelerator.

    Force use the cpu on node has both cpu and gpu: `INC_TARGET_DEVICE=cpu` python main.py ...
    The `INC_TARGET_DEVICE` is case insensitive.
    The environment variable `INC_TARGET_DEVICE` has higher priority than the `device_name`.
    TODO: refine the docs and logic later
    """
    # 1. Get the device setting from environment variable `INC_TARGET_DEVICE`.
    INC_TARGET_DEVICE = os.environ.get("INC_TARGET_DEVICE", None)
    if INC_TARGET_DEVICE:
        INC_TARGET_DEVICE = INC_TARGET_DEVICE.lower()
    # 2. If the `INC_TARGET_DEVICE` is set and the accelerator is available, use it.
    if INC_TARGET_DEVICE and accelerator_registry.get_accelerator_cls_by_name(INC_TARGET_DEVICE) is not None:
        logger.warning("Force use %s accelerator.", INC_TARGET_DEVICE)
        return accelerator_registry.get_accelerator_cls_by_name(INC_TARGET_DEVICE)()
    # 3. If the `device_name` is set and the accelerator is available, use it.
    if device_name != "auto":
        if accelerator_registry.get_accelerator_cls_by_name(device_name) is not None:
            accelerator_cls = accelerator_registry.get_accelerator_cls_by_name(device_name)
            logger.warning("Selected accelerator %s by device_name.", accelerator_cls.__name__)
            return accelerator_cls()
        else:
            logger.warning("The device name %s is not supported, use auto detect instead.", device_name)
    # 4. Select the accelerator by priority.
    for accelerator_cls in accelerator_registry.get_sorted_accelerators():
        if accelerator_cls.is_available():
            logger.warning("Auto detect accelerator: %s.", accelerator_cls.__name__)
            accelerator = accelerator_cls()
            return accelerator


# Force use cpu accelerator even if cuda is available.
# INC_TARGET_DEVICE = "cpu" python ...
# or
# INC_TARGET_DEVICE = "CPU" python ...
# or
# CUDA_VISIBLE_DEVICES="" python ...
