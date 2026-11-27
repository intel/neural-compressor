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
"""Quantized module registry, target pattern matching, and in-place model conversion."""

from __future__ import annotations

import fnmatch
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type

import torch
import torch.nn as nn

from neural_compressor.common.utils import FAKE_QUANT, Mode
from neural_compressor.torch.utils import logger, register_algo

from .calib import calibrate
from .config import FakeQuantConfig, ModuleQuantConfig
from .save_load import (
    apply_amax_map_to_module_config,
    canonicalize_module_name,
    load_amax_map_from_state,
)
from .tensor_quantizer import TensorQuantizer

# ---------------------------------------------------------------------------
# Quantized Module Registry (QuantModuleRegistry)
# ---------------------------------------------------------------------------

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


def _ensure_builtin_modules_registered() -> None:
    if nn.Linear not in _MODULE_REGISTRY:
        from . import quant_linear  # noqa: F401


def _lookup_factory(mod_cls: Type[nn.Module]) -> Optional[QuantFactory]:
    _ensure_builtin_modules_registered()
    if getattr(mod_cls, "_is_fake_quant_module", False):
        return None
    if mod_cls.__module__.startswith("peft.") and mod_cls not in _MODULE_REGISTRY:
        from .plugins.peft import register_peft_quant_modules

        register_peft_quant_modules()
    if mod_cls not in _MODULE_REGISTRY:
        from .plugins.moe import is_fused_moe_experts_class, register_moe_quant_modules

        if is_fused_moe_experts_class(mod_cls):
            register_moe_quant_modules()
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
    if _lookup_factory(type(module)) is not None:
        return True
    from .plugins.moe import _MOE_REGISTERED_CLASSES, QuantFusedMoEExperts, is_fused_moe_experts_module

    if is_fused_moe_experts_module(module):
        _MOE_REGISTERED_CLASSES.add(type(module))
        register_quant_module(type(module), QuantFusedMoEExperts.from_float)
        return True
    return False


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
    _ensure_builtin_modules_registered()
    return tuple(_MODULE_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Target Pattern Matching & Model Conversion
# ---------------------------------------------------------------------------

_LINEAR_LIKE_TYPE_NAMES = frozenset({"Linear", "QuantLinear", "QuantLoraLinear"})


def _op_type_name(op_type: Any) -> str:
    if isinstance(op_type, str):
        return op_type
    if isinstance(op_type, type):
        return op_type.__name__
    return type(op_type).__name__


def _matches_single_name(module_name: str, op_type: Any, pattern: str) -> bool:
    if pattern.startswith("re:"):
        return re.fullmatch(pattern[3:], module_name) is not None

    type_name = _op_type_name(op_type)
    if pattern == "Linear":
        if type_name in _LINEAR_LIKE_TYPE_NAMES or type_name == "QuantFusedMoEExperts":
            return True
        if isinstance(op_type, type) and issubclass(op_type, nn.Module):
            if issubclass(op_type, nn.Linear):
                return True
            from .plugins.moe import is_fused_moe_experts_class

            if is_fused_moe_experts_class(op_type):
                return True
        if "Linear" in type_name:
            return True
    elif pattern == "FusedMoEExperts":
        if type_name == "QuantFusedMoEExperts" or type_name.endswith("Experts"):
            return True
        if isinstance(op_type, type) and issubclass(op_type, nn.Module):
            from .plugins.moe import is_fused_moe_experts_class

            if is_fused_moe_experts_class(op_type):
                return True
    elif pattern == type_name:
        return True

    if any(ch in pattern for ch in "*?["):
        return fnmatch.fnmatchcase(module_name, pattern)

    return module_name == pattern or module_name.endswith(f".{pattern}")


def matches_target(module_name: str, op_type: Any, pattern: str) -> bool:
    """Match ``(module_name, op_type)`` against a ``compressed-tensors`` / ``ModelOpt`` target pattern."""
    if _matches_single_name(module_name, op_type, pattern):
        return True
    canonical = canonicalize_module_name(module_name)
    if canonical != module_name:
        return _matches_single_name(canonical, op_type, pattern)
    return False


def _has_proper_ancestor(name: str, roots: set[str]) -> bool:
    """Return True in ``O(depth)`` if any proper dot-separated prefix of ``name`` is in ``roots``."""
    if not roots:
        return False
    idx = name.find(".")
    while idx != -1:
        if name[:idx] in roots:
            return True
        idx = name.find(".", idx + 1)
    return False


@dataclass
class QuantConversionReport:
    """Resolved per-module fake-quantization mapping and conversion audit statistics."""

    plan: Dict[str, ModuleQuantConfig] = field(default_factory=dict)
    ignored: List[str] = field(default_factory=list)
    unmatched: List[str] = field(default_factory=list)
    op_types: Dict[str, Any] = field(default_factory=dict)

    @property
    def module_configs(self) -> Dict[str, ModuleQuantConfig]:
        """Idiomatic accessor for resolved ``{module_name: ModuleQuantConfig}``."""
        return self.plan

    def summary(self) -> dict[str, Any]:
        """Return structured counts by ``(weight_fmt, input_fmt, scheme)``."""
        combo_counts: Counter[str] = Counter()
        pending_static = 0
        for cfg in self.plan.values():
            w_desc = "bf16" if cfg.weights.is_noop else f"{cfg.weights.format.name}({cfg.weights.scheme})"
            i_desc = (
                "bf16"
                if cfg.input_activations.is_noop
                else f"{cfg.input_activations.format.name}({cfg.input_activations.scheme})"
            )
            combo_counts[f"W={w_desc}, A={i_desc}"] += 1
            for q_cfg in (cfg.weights, cfg.input_activations, cfg.output_activations):
                if not q_cfg.is_noop and not q_cfg.dynamic and q_cfg.amax is None:
                    pending_static += 1
        return {
            "quantized_modules": len(self.plan),
            "ignored_modules": len(self.ignored),
            "unmatched_modules": len(self.unmatched),
            "pending_static_quantizers": pending_static,
            "schemes": dict(combo_counts),
        }

    def format_table(self) -> str:
        """Render a concise human-readable conversion summary."""
        stats = self.summary()
        lines = [
            f"QuantConversionReport: quantized={stats['quantized_modules']}, "
            f"ignored={stats['ignored_modules']}, unmatched={stats['unmatched_modules']}, "
            f"pending_static={stats['pending_static_quantizers']}"
        ]
        for scheme_label, count in sorted(stats["schemes"].items()):
            lines.append(f"  - {scheme_label}: {count} modules")
        return "\n".join(lines)


PlanReport = QuantConversionReport


def match_quant_config_from_info(
    model_info: Iterable[Tuple[str, Any]],
    config: FakeQuantConfig,
) -> QuantConversionReport:
    """Resolve ``config`` against ``(module_name, op_type)`` tuples without mutating a model."""
    state_map: Optional[dict[str, float]] = None
    if config.quantizer_state_path is not None:
        state_map = load_amax_map_from_state(config.quantizer_state_path)

    entries = list(model_info)
    parent_roots = {name for name, _ in entries}
    filtered_entries = [(name, op_type) for name, op_type in entries if not _has_proper_ancestor(name, parent_roots)]

    report = QuantConversionReport()
    for module_name, op_type in filtered_entries:
        report.op_types[module_name] = op_type
        if any(matches_target(module_name, op_type, pat) for pat in config.ignore):
            report.ignored.append(module_name)
            continue

        resolved_cfg: Optional[ModuleQuantConfig] = None
        for rule in config.config_groups:
            if any(matches_target(module_name, op_type, pat) for pat in rule.targets):
                resolved_cfg = rule.resolve(default_backend=config.backend)

        if resolved_cfg is None or resolved_cfg.is_noop:
            report.unmatched.append(module_name)
            continue

        if state_map is not None:
            apply_amax_map_to_module_config(module_name, resolved_cfg, state_map)

        report.plan[module_name] = resolved_cfg
    return report


compile_plan_from_info = match_quant_config_from_info


def _collect_supported_modules(root: nn.Module) -> List[Tuple[str, Any]]:
    """Traverse ``root`` top-down, stopping at atomic fake-quant or supported leaf/wrapper modules."""
    collected: List[Tuple[str, Any]] = []

    def _walk(mod: nn.Module, prefix: str) -> None:
        for child_name, child in mod._modules.items():
            if child is None:
                continue
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if getattr(child, "_is_fake_quant_module", False):
                continue
            if is_supported_module(child):
                collected.append((full_name, type(child)))
                # Do not recurse into internal submodules (e.g. LoraLinear.base_layer / lora_A / lora_B).
                continue
            _walk(child, full_name)

    _walk(root, "")
    return collected


def match_quant_config(model: nn.Module, config: FakeQuantConfig) -> QuantConversionReport:
    """Inspect ``model`` and produce a :class:`QuantConversionReport` for ``config``."""
    info = _collect_supported_modules(model)
    return match_quant_config_from_info(info, config)


build_quant_plan = match_quant_config


def _set_submodule(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    if "." in dotted_name:
        parent_name, child_name = dotted_name.rsplit(".", 1)
        parent = root.get_submodule(parent_name)
    else:
        parent = root
        child_name = dotted_name
    setattr(parent, child_name, new_module)


def replace_quant_modules(model: nn.Module, report: QuantConversionReport) -> dict[str, int]:
    """Replace every matched module in ``model`` with its registered fake-quant counterpart."""
    n_replaced = 0
    for module_name, module_cfg in report.plan.items():
        if module_cfg.is_noop:
            continue
        target = model.get_submodule(module_name)
        replacement = quantize_module(target, module_cfg)
        if replacement is not target:
            _set_submodule(model, module_name, replacement)
            n_replaced += 1
    return {
        "n_replaced": n_replaced,
        "n_ignored": len(report.ignored),
        "n_unmatched": len(report.unmatched),
    }


apply_quant_plan = replace_quant_modules


def convert_to_quantized_model(model: nn.Module, config: FakeQuantConfig) -> QuantConversionReport:
    """Match ``config`` against ``model`` and replace matched modules in-place."""
    report = match_quant_config(model, config)
    replace_quant_modules(model, report)
    return report


# ---------------------------------------------------------------------------
# INC Algorithm Entry (@register_algo("fake_quant"))
# ---------------------------------------------------------------------------

_QUANTIZER_SLOTS = (
    ("weights", "weight_quantizer"),
    ("input_activations", "input_quantizer"),
    ("output_activations", "output_quantizer"),
)


def _pending_static(plan: Dict[str, ModuleQuantConfig]) -> list[str]:
    pending = []
    for name, config in plan.items():
        for attr_name, slot_name in _QUANTIZER_SLOTS:
            quantizer_cfg = getattr(config, attr_name)
            if quantizer_cfg is None or quantizer_cfg.is_noop or quantizer_cfg.dynamic:
                continue
            if quantizer_cfg.amax is None:
                pending.append(f"{name}.{slot_name}")
    return pending


def _set_calibrating(model: torch.nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, TensorQuantizer) and not module.is_dynamic and module.is_enabled:
            if enabled:
                if module.amax is None:
                    module.enable_calib()
                else:
                    module.disable_calib()
            else:
                module.disable_calib()


@register_algo(name=FAKE_QUANT)
def fake_quant_entry(
    model: torch.nn.Module,
    configs_mapping,
    mode: Mode = Mode.QUANTIZE,
    run_fn=None,
    run_args: Any = None,
    **kwargs,
) -> torch.nn.Module:
    """Replace modules with fake-quant subclasses and optionally calibrate.

    Args:
        model: Float model, modified in place.
        configs_mapping: ``{(module_name, op_type): ModuleQuantConfig}`` from
            :meth:`FakeQuantConfig.to_config_mapping`.
        mode: ``PREPARE`` inserts quantizers only; ``QUANTIZE`` also calibrates
            static quantizers when ``run_fn`` is given; ``CONVERT`` finalizes
            calibration without replacing the fake-quant modules.
        run_fn: Calibration forward loop, called as ``run_fn(model, *run_args)``.
        run_args: Positional arguments for ``run_fn``.
    """
    if mode == Mode.CONVERT:
        for module in model.modules():
            if isinstance(module, TensorQuantizer):
                module.finalize()
        logger.info("[fake_quant] finalized calibration; export() produces deployment artifacts")
        return model

    plan = {name: config for (name, _op_type), config in configs_mapping.items()}
    report = QuantConversionReport(
        plan=plan,
        op_types={name: op_type for (name, op_type) in configs_mapping},
    )
    pending = _pending_static(plan)
    if mode != Mode.PREPARE and pending and run_fn is None:
        head = ", ".join(pending[:5])
        raise ValueError(
            f"{len(pending)} static quantizers have no amax and no run_fn was given: {head} ...; "
            "pass run_fn/run_args, an amax summary, or use prepare()+calibrate()"
        )
    result = replace_quant_modules(model, report)
    logger.info(f"[fake_quant] replaced {result['n_replaced']} modules with quantized subclasses")

    if mode == Mode.PREPARE:
        _set_calibrating(model, True)
        if pending:
            logger.info(f"[fake_quant] {len(pending)} static quantizers await calibration")
        return model

    if pending:
        logger.info(f"[fake_quant] calibrating {len(pending)} static quantizers with run_fn")
        calibrate(model, run_fn, run_args or ())
    for module in model.modules():
        if isinstance(module, TensorQuantizer):
            module.finalize()
    return model


__all__ = [
    "PlanReport",
    "QuantConversionReport",
    "QuantFactory",
    "apply_quant_plan",
    "build_quant_plan",
    "compile_plan_from_info",
    "convert_to_quantized_model",
    "fake_quant_entry",
    "is_supported_module",
    "match_quant_config",
    "match_quant_config_from_info",
    "matches_target",
    "quantize_module",
    "register_quant_module",
    "replace_quant_modules",
    "supported_module_types",
    "unregister_quant_module",
]
