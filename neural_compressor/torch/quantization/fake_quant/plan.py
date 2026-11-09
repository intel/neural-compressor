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
"""User-facing quantization rules and the compiled per-module plan."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from neural_compressor.torch.utils import logger

from .modules import QuantLinear
from .options import ModuleQuantConfig, QuantizerConfig
from .registry import QuantModuleRegistry

DEFAULT_EXCLUDE = (r"^lm_head$", r".*embed_tokens.*")

_QUANTIZER_FIELDS = ("weight", "input", "output")


@dataclass
class QuantRule:
    """One declarative rule: match modules, then configure their quantizers.

    Exactly one of ``module`` / ``quantizer`` selects the target:

    * ``module``: ``re.search`` against the qualified module name
      (e.g. ``.*\\.mlp\\.(gate_proj|up_proj|down_proj)$``). This is the common
      case: recipe authors know layer names, not internal quantizer names.
    * ``quantizer``: ``re.search`` against ``<module>.<quantizer_name>``
      (e.g. ``.*mlp.*weight_quantizer$``). Escape hatch for per-quantizer control.

    ``weight`` / ``input`` / ``output`` accept a format name, a dict or a
    :class:`QuantizerConfig`. Rules are applied in list order; later rules
    override earlier ones for the quantizers they set.
    """

    module: Optional[str] = None
    quantizer: Optional[str] = None
    parent_class: Optional[str] = None
    weight: Optional[QuantizerConfig | str | dict] = None
    input: Optional[QuantizerConfig | str | dict] = None
    output: Optional[QuantizerConfig | str | dict] = None
    enable: bool = True

    def __post_init__(self) -> None:
        """Validate the selector and coerce the quantizer configs."""
        if (self.module is None) == (self.quantizer is None):
            raise ValueError("QuantRule needs exactly one of `module=` or `quantizer=`")
        for pattern in (self.module, self.quantizer):
            if pattern is not None:
                try:
                    re.compile(pattern)
                except re.error as exc:
                    raise ValueError(f"invalid regex {pattern!r}: {exc}") from exc
        self.weight = QuantizerConfig.from_any(self.weight)
        self.input = QuantizerConfig.from_any(self.input)
        self.output = QuantizerConfig.from_any(self.output)
        if not self.enable:
            self.weight = self.input = self.output = QuantizerConfig(format="bf16")

    def matches(self, module_name: str, op_type: str) -> bool:
        """Return True when the rule targets this module."""
        if self.parent_class is not None and op_type != self.parent_class:
            return False
        if self.module is not None:
            return re.search(self.module, module_name) is not None
        prefix = f"{module_name}." if module_name else ""
        return any(re.search(self.quantizer, f"{prefix}{field}_quantizer") is not None for field in _QUANTIZER_FIELDS)

    def contributions(self, module_name: str, op_type: str) -> Dict[str, QuantizerConfig]:
        """Return the quantizer configs this rule sets for a matched module."""
        if self.quantizer is None:
            return {f: getattr(self, f) for f in _QUANTIZER_FIELDS if getattr(self, f) is not None}
        prefix = f"{module_name}." if module_name else ""
        out: Dict[str, QuantizerConfig] = {}
        for field_name in _QUANTIZER_FIELDS:
            if getattr(self, field_name) is None:
                continue
            if re.search(self.quantizer, f"{prefix}{field_name}_quantizer") is not None:
                out[field_name] = getattr(self, field_name)
        return out

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dictionary."""
        payload = {"enable": self.enable}
        for key in ("module", "quantizer", "parent_class"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        for key in _QUANTIZER_FIELDS:
            value = getattr(self, key)
            payload[key] = value.to_dict() if value is not None else None
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "QuantRule":
        """Build a rule from :meth:`to_dict` output."""
        return cls(**payload)


@dataclass
class PlanReport:
    """What :func:`compile_plan` did, for logging and tests."""

    plan: Dict[str, ModuleQuantConfig]
    op_types: Dict[str, str] = field(default_factory=dict)
    rule_hits: list[int] = field(default_factory=list)
    unmatched_rules: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)

    @property
    def n_modules(self) -> int:
        """Number of modules that will be quantized."""
        return len(self.plan)


def _supported_type_names() -> set[str]:
    return {src.__name__ for src in QuantModuleRegistry._registry}


def _is_empty(config: ModuleQuantConfig) -> bool:
    return all(getattr(config, field_name) is None for field_name in _QUANTIZER_FIELDS)


def _compile_items(
    items: Sequence[Tuple[str, str]],
    rules: Iterable[QuantRule],
    *,
    default: Optional[ModuleQuantConfig],
    exclude: Iterable[str],
    strict: bool,
) -> PlanReport:
    rules = list(rules)
    exclude_patterns = tuple(exclude)
    supported = _supported_type_names()
    report = PlanReport(plan={}, rule_hits=[0] * len(rules))
    for name, op_type in items:
        if not name or op_type not in supported:
            continue
        if any(re.search(pattern, name) for pattern in exclude_patterns):
            report.excluded.append(name)
            continue
        config = ModuleQuantConfig()
        if default is not None:
            config = ModuleQuantConfig(weight=default.weight, input=default.input, output=default.output)
        for index, rule in enumerate(rules):
            if not rule.matches(name, op_type):
                continue
            report.rule_hits[index] += 1
            for field_name, quantizer_cfg in rule.contributions(name, op_type).items():
                # Copy: the same rule config is shared by every matched module,
                # but calibrated amax is per-module state.
                setattr(config, field_name, copy.copy(quantizer_cfg))
        if not _is_empty(config):
            report.plan[name] = config
            report.op_types[name] = op_type
    for index, rule in enumerate(rules):
        if report.rule_hits[index] == 0:
            selector = rule.module if rule.module is not None else rule.quantizer
            report.unmatched_rules.append(selector)
    if strict and report.unmatched_rules:
        raise ValueError(
            "these quantization rules matched 0 modules: "
            f"{report.unmatched_rules}; check the regex or the model module names"
        )
    return report


def compile_plan(
    model: nn.Module,
    rules: Iterable[QuantRule],
    *,
    default: Optional[ModuleQuantConfig] = None,
    exclude: Iterable[str] = DEFAULT_EXCLUDE,
    strict: bool = True,
) -> PlanReport:
    """Match rules against a live model and return one config per module."""
    items = [
        (name, type(module).__name__)
        for name, module in model.named_modules()
        if name and not isinstance(module, QuantLinear)
    ]
    return _compile_items(items, rules, default=default, exclude=exclude, strict=strict)


def compile_plan_from_info(
    model_info: Sequence[Tuple[str, str]],
    rules: Iterable[QuantRule],
    *,
    default: Optional[ModuleQuantConfig] = None,
    exclude: Iterable[str] = DEFAULT_EXCLUDE,
    strict: bool = True,
) -> PlanReport:
    """Compile a plan from ``(name, type_name)`` pairs (NC's ``model_info``)."""
    return _compile_items(model_info, rules, default=default, exclude=exclude, strict=strict)


def summarize_plan(report: PlanReport, model: Optional[nn.Module] = None) -> dict:
    """Aggregate a plan by ``format/scheme`` for logging and tests."""
    stats: Dict[str, dict] = {}
    params = dict(model.named_parameters()) if model is not None else {}
    for name, config in report.plan.items():
        for field_name in _QUANTIZER_FIELDS:
            quantizer_cfg = getattr(config, field_name)
            if quantizer_cfg is None or quantizer_cfg.is_noop:
                continue
            key = f"{quantizer_cfg.format.name}/{quantizer_cfg.scheme}"
            bucket = stats.setdefault(key, {"n_quantizers": 0, "n_params": 0})
            bucket["n_quantizers"] += 1
            if field_name == "weight":
                bucket["n_params"] += params.get(f"{name}.weight", torch.empty(0)).numel()
    return stats


def print_plan_summary(report: PlanReport, model: Optional[nn.Module] = None, tag: str = "plan") -> None:
    """Log a one-line-per-format summary plus rule statistics."""
    stats = summarize_plan(report, model)
    parts = [
        f"{key}={value['n_quantizers']} quantizers"
        + (f" ({value['n_params'] / 1e9:.2f}B params)" if value["n_params"] else "")
        for key, value in sorted(stats.items())
    ]
    logger.info(f"[{tag}] {report.n_modules} modules quantized: " + (", ".join(parts) if parts else "none"))
    if report.excluded:
        logger.info(f"[{tag}] excluded {len(report.excluded)} modules (lm_head/embed_tokens)")
    for index, hits in enumerate(report.rule_hits):
        logger.debug(f"[{tag}] rule[{index}] hits={hits}")


def apply_plan(model: nn.Module, report: PlanReport, *, share_params: bool = False) -> dict:
    """Replace every module in ``report.plan`` with its quantized subclass."""
    replaced = []
    for name, config in report.plan.items():
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        module = getattr(parent, leaf)
        target_cls = QuantModuleRegistry.get(module)
        if target_cls is None:
            raise TypeError(f"no quantized class for {name} ({type(module).__name__})")
        quantized = target_cls.from_float(module, config, share_params=share_params)
        setattr(parent, leaf, quantized)
        replaced.append({"name": name})
    return {"replaced": replaced, "n_replaced": len(replaced)}


__all__ = [
    "DEFAULT_EXCLUDE",
    "PlanReport",
    "QuantRule",
    "apply_plan",
    "compile_plan",
    "compile_plan_from_info",
    "print_plan_summary",
    "summarize_plan",
]
