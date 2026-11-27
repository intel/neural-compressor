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
"""Extensible registry mapping source ``nn.Module`` classes to fake-quantized replacements."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple, Type

import torch.nn as nn

from .config import ModuleQuantConfig

QuantFactory = Callable[[nn.Module, ModuleQuantConfig], nn.Module]

_MODULE_REGISTRY: Dict[Type[nn.Module], QuantFactory] = {}


def register_quant_module(
    source_cls: Type[nn.Module],
    factory_fn: Optional[QuantFactory] = None,
):
    """Register ``factory_fn`` (or class exposing ``.from_float(mod, cfg)``) for ``source_cls``.

    Can be used either as a decorator ``@register_quant_module(nn.Linear)`` or a direct function call
    ``register_quant_module(LoraLinear, QuantLoraLinear.from_lora)``.
    """
    if factory_fn is not None:
        _MODULE_REGISTRY[source_cls] = factory_fn
        return factory_fn

    def _decorator(quant_cls):
        if hasattr(quant_cls, "from_float") and callable(quant_cls.from_float):
            _MODULE_REGISTRY[source_cls] = quant_cls.from_float
        elif callable(quant_cls):
            _MODULE_REGISTRY[source_cls] = quant_cls
        else:
            raise TypeError(f"{quant_cls!r} must be callable or define .from_float(mod, config)")
        return quant_cls

    return _decorator


def unregister_quant_module(source_cls: Type[nn.Module]) -> None:
    """Remove ``source_cls`` from the fake-quant module registry if present."""
    _MODULE_REGISTRY.pop(source_cls, None)


def _lookup_factory(mod_cls: Type[nn.Module]) -> Optional[QuantFactory]:
    if getattr(mod_cls, "_is_fake_quant_module", False):
        return None
    if mod_cls.__module__.startswith("peft.") and mod_cls not in _MODULE_REGISTRY:
        from .plugins.peft import register_peft_quant_modules

        register_peft_quant_modules()
    exact = _MODULE_REGISTRY.get(mod_cls)
    if exact is not None:
        return exact
    for base in mod_cls.__mro__[1:]:
        if issubclass(base, nn.Module) and base in _MODULE_REGISTRY:
            return _MODULE_REGISTRY[base]
    return None


def is_supported_module(module: nn.Module) -> bool:
    """Return True if ``module`` is an unwrapped module with a registered fake-quant replacement."""
    if getattr(module, "_is_fake_quant_module", False):
        return False
    return _lookup_factory(type(module)) is not None


def quantize_module(module: nn.Module, config: ModuleQuantConfig) -> nn.Module:
    """Convert ``module`` using the registered factory for ``type(module)``."""
    if config.is_noop or getattr(module, "_is_fake_quant_module", False):
        return module
    factory = _lookup_factory(type(module))
    if factory is None:
        raise KeyError(
            f"No fake-quant replacement registered for {type(module).__name__}; "
            f"registered types: {[cls.__name__ for cls in _MODULE_REGISTRY]}"
        )
    return factory(module, config)


def supported_module_types() -> Tuple[Type[nn.Module], ...]:
    """Return all registered source ``nn.Module`` types."""
    return tuple(_MODULE_REGISTRY.keys())


__all__ = [
    "QuantFactory",
    "is_supported_module",
    "quantize_module",
    "register_quant_module",
    "supported_module_types",
    "unregister_quant_module",
]
