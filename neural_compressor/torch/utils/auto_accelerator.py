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


# To keep it simply, only add the APIs we need.

import os
from abc import ABC, abstractmethod
from typing import Any, Callable, List

import torch

from neural_compressor.common.utils import LazyImport
from neural_compressor.torch.utils import logger

htcore = LazyImport("habana_frameworks.torch.core")

PRIORITY_HPU = 100
PRIORITY_CUDA = 95
PRIORITY_CPU = 90


class AcceleratorRegistry:
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


class Auto_Accelerator(ABC):
    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        pass

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def device_name(self, device_indx) -> str:
        pass

    @abstractmethod
    def set_device(self, device_index):
        pass

    @abstractmethod
    def current_device(self):
        pass

    @abstractmethod
    def current_device_name(self):
        pass

    @abstractmethod
    def device(self, device_index=None):
        pass

    @abstractmethod
    def empty_cache(self):
        pass

    @abstractmethod
    def synchronize(self):
        pass

    @abstractmethod
    def mark_step(self):
        pass


@register_accelerator(name="cpu", priority=PRIORITY_CPU)
class CPU_Accelerator(Auto_Accelerator):
    def __init__(self) -> None:
        self._name = "cpu"

    def name(self) -> str:
        return self._name

    @classmethod
    def is_available(cls) -> bool:
        return True

    def device_name(self, device_indx) -> str:
        return "cpu"

    def set_device(self, device_index):
        pass

    def current_device(self):
        return "cpu"

    def current_device_name(self):
        return "cpu"

    def device(self, device_index=None):
        pass

    def empty_cache(self):
        pass

    def synchronize(self):
        pass

    def mark_step(self):
        pass


@register_accelerator(name="cuda", priority=PRIORITY_CUDA)
class CUDA_Accelerator(Auto_Accelerator):
    def __init__(self) -> None:
        self._name = "cuda"

    def name(self) -> str:
        return self._name

    @classmethod
    def is_available(cls) -> bool:
        return torch.cuda.is_available()

    def device_name(self, device_indx) -> str:
        if device_indx is None:
            return "cuda"
        return f"cuda:{device_indx}"

    def synchronize(self):
        return torch.cuda.synchronize()

    def set_device(self, device_index):
        return torch.cuda.set_device(device_index)

    def current_device(self):
        return torch.cuda.current_device()

    def current_device_name(self):
        return "cuda:{}".format(torch.cuda.current_device())

    def device(self, device_index=None):
        return torch.cuda.device(device_index)

    def empty_cache(self):
        return torch.cuda.empty_cache()

    def mark_step(self):
        pass


@register_accelerator(name="hpu", priority=PRIORITY_HPU)
class HPU_Accelerator(Auto_Accelerator):
    def __init__(self) -> None:
        self._name = "hpu"

    def name(self) -> str:
        return self._name

    @classmethod
    def is_available(cls) -> bool:
        from .environ import is_hpex_available

        if is_hpex_available():
            return torch.hpu.is_available()
        else:
            return False

    def device_name(self, device_indx) -> str:
        if device_indx is None:
            return "hpu"
        return f"hpu:{device_indx}"

    def synchronize(self):
        return torch.hpu.synchronize()

    def set_device(self, device_index):
        return torch.hpu.set_device(device_index)

    def current_device(self):
        return torch.hpu.current_device()

    def current_device_name(self):
        return "hpu:{}".format(torch.hpu.current_device())

    def device(self, device_index=None):
        return torch.hpu.device(device_index)

    def empty_cache(self):
        return torch.hpu.empty_cache()

    def mark_step(self):
        return htcore.mark_step()


def auto_detect_accelerator(device_name="auto") -> Auto_Accelerator:
    # Force use the cpu on node has both cpu and gpu: `FORCE_DEVICE=cpu` python main.py ...
    # The `FORCE_DEVICE` is case insensitive.
    # The environment variable `FORCE_DEVICE` has higher priority than the `device_name`.
    # TODO: refine the docs and logic later
    # 1. Get the device setting from environment variable `FORCE_DEVICE`.
    FORCE_DEVICE = os.environ.get("FORCE_DEVICE", None)
    if FORCE_DEVICE:
        FORCE_DEVICE = FORCE_DEVICE.lower()
    # 2. If the `FORCE_DEVICE` is set and the accelerator is available, use it.
    if FORCE_DEVICE and accelerator_registry.get_accelerator_cls_by_name(FORCE_DEVICE) is not None:
        logger.warning("Force use %s accelerator.", FORCE_DEVICE)
        return accelerator_registry.get_accelerator_cls_by_name(FORCE_DEVICE)()
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
# FORCE_DEVICE = "cpu" python ...
# or
# FORCE_DEVICE = "CPU" python ...
# or
# CUDA_VISIBLE_DEVICES="" python ...
