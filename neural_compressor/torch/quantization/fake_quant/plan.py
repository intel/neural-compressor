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
"""Quantization plan resolution, reporting, and in-place module replacement."""

from __future__ import annotations

import fnmatch
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from .calibrate import (
    apply_amax_map_to_module_config,
    canonicalize_module_name,
    load_amax_map_from_state,
)
from .config import FakeQuantConfig, ModuleQuantConfig
from .registry import is_supported_module, quantize_module

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
        if type_name in _LINEAR_LIKE_TYPE_NAMES:
            return True
        if isinstance(op_type, type) and issubclass(op_type, nn.Linear):
            return True
        if "Linear" in type_name:
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
class PlanReport:
    """Resolved per-module fake-quantization plan and audit statistics."""

    plan: Dict[str, ModuleQuantConfig] = field(default_factory=dict)
    ignored: List[str] = field(default_factory=list)
    unmatched: List[str] = field(default_factory=list)
    op_types: Dict[str, Any] = field(default_factory=dict)

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
        """Render a concise human-readable plan summary."""
        stats = self.summary()
        lines = [
            f"FakeQuantPlan: quantized={stats['quantized_modules']}, "
            f"ignored={stats['ignored_modules']}, unmatched={stats['unmatched_modules']}, "
            f"pending_static={stats['pending_static_quantizers']}"
        ]
        for scheme_label, count in sorted(stats["schemes"].items()):
            lines.append(f"  - {scheme_label}: {count} modules")
        return "\n".join(lines)


def compile_plan_from_info(
    model_info: Iterable[Tuple[str, Any]],
    config: FakeQuantConfig,
) -> PlanReport:
    """Resolve ``config`` against ``(module_name, op_type)`` tuples without mutating a model."""
    state_map: Optional[dict[str, float]] = None
    if config.quantizer_state_path is not None:
        state_map = load_amax_map_from_state(config.quantizer_state_path)

    entries = list(model_info)
    parent_roots = {name for name, _ in entries}
    filtered_entries = [(name, op_type) for name, op_type in entries if not _has_proper_ancestor(name, parent_roots)]

    report = PlanReport()
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


def build_quant_plan(model: nn.Module, config: FakeQuantConfig) -> PlanReport:
    """Inspect ``model`` and produce a :class:`PlanReport` for ``config``."""
    info = _collect_supported_modules(model)
    return compile_plan_from_info(info, config)


def _set_submodule(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    if "." in dotted_name:
        parent_name, child_name = dotted_name.rsplit(".", 1)
        parent = root.get_submodule(parent_name)
    else:
        parent = root
        child_name = dotted_name
    setattr(parent, child_name, new_module)


def apply_quant_plan(model: nn.Module, report: PlanReport) -> dict[str, int]:
    """Replace every planned module in ``model`` with its registered fake-quant counterpart."""
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


__all__ = [
    "PlanReport",
    "apply_quant_plan",
    "build_quant_plan",
    "compile_plan_from_info",
    "matches_target",
]
