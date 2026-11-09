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
"""Registry mapping float module classes to their quantized subclasses."""

from __future__ import annotations

from typing import Dict, Optional

import torch.nn as nn

from .options import ModuleQuantConfig


class QuantModuleRegistry:
    """Map ``nn.Linear`` -> ``QuantLinear`` and convert instances in place.

    The replacement is a real subclass (ModelOpt's approach) rather than
    rebinding ``module.__class__``: ``isinstance``, pickling, ``torch.compile``
    and FSDP2 all rely on the actual type.
    """

    _registry: Dict[type, type] = {}

    @classmethod
    def register(cls, src: type, dst: Optional[type] = None):
        """Register ``dst`` for ``src``; usable directly or as a decorator."""
        if dst is not None:
            cls._registry[src] = dst
            return dst

        def decorator(target: type) -> type:
            cls._registry[src] = target
            return target

        return decorator

    @classmethod
    def get(cls, module_or_type) -> Optional[type]:
        """Return the quantized class for a module or class, walking the MRO."""
        module_type = module_or_type if isinstance(module_or_type, type) else type(module_or_type)
        for candidate in module_type.__mro__:
            if candidate in cls._registry:
                return cls._registry[candidate]
        return None

    @classmethod
    def convert(cls, module: nn.Module, config: ModuleQuantConfig) -> nn.Module:
        """Build a quantized copy of ``module``; raises for unsupported types."""
        target = cls.get(module)
        if target is None:
            raise TypeError(
                f"no quantized implementation registered for {type(module).__name__}; "
                f"register it with QuantModuleRegistry.register({type(module).__name__}, ...)"
            )
        return target.from_float(module, config)

    @classmethod
    def registered_types(cls) -> Dict[str, str]:
        """Return a ``{float class: quantized class}`` name mapping."""
        return {
            f"{src.__module__}.{src.__name__}": f"{dst.__module__}.{dst.__name__}" for src, dst in cls._registry.items()
        }


__all__ = ["QuantModuleRegistry"]
