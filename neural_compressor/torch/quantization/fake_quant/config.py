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
"""``FakeQuantConfig``: the whole-model policy consumed by ``quantize()``."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Tuple, Union

from neural_compressor.common.base_config import register_config
from neural_compressor.common.utils import EMPTY_WHITE_LIST, FAKE_QUANT
from neural_compressor.torch.quantization.config import TorchBaseConfig
from neural_compressor.torch.utils import logger

from .calibrator import AmaxMap, load_amax, modelopt_megatron_translate
from .options import ModuleQuantConfig, QuantizerConfig
from .plan import DEFAULT_EXCLUDE, PlanReport, QuantRule, compile_plan_from_info


def _coerce_module_config(value: Any) -> Optional[ModuleQuantConfig]:
    if value is None or isinstance(value, ModuleQuantConfig):
        return value
    if isinstance(value, dict):
        return ModuleQuantConfig(
            weight=QuantizerConfig.from_any(value.get("weight")),
            input=QuantizerConfig.from_any(value.get("input")),
            output=QuantizerConfig.from_any(value.get("output")),
        )
    raise TypeError(f"cannot build ModuleQuantConfig from {type(value)!r}")


@register_config(framework_name="torch", algo_name=FAKE_QUANT)
class FakeQuantConfig(TorchBaseConfig):
    """Insert fake-quant modules into a model; shared by QAT and QAD.

    The config is a list of ordered :class:`QuantRule` entries plus a default
    policy. It compiles to NC's per-op mapping, so ``quantize()`` dispatches it
    through ``algos_mapping["fake_quant"]`` like any other algorithm.

    Args:
        rules: Ordered rules; later rules override earlier ones per quantizer.
        default: Baseline :class:`ModuleQuantConfig` for supported modules.
        algorithm: Calibration algorithm for static quantizers. Only ``"max"``
            (per-tensor max absolute value over the calibration batches) is
            implemented; it is unused when ``amax_source`` supplies values.
        exclude: Regexes for modules that must never be quantized.
        amax_source: Where static amax values come from. ``str``/``Path`` or
            ``{"file": path, "translate": name}``.
        strict: Raise when a rule matches zero modules.
        white_list: Kept for ``BaseConfig`` compatibility; unused.
    """

    name = FAKE_QUANT

    def __init__(
        self,
        rules: Optional[Iterable[Union[QuantRule, dict]]] = None,
        default: Optional[Union[ModuleQuantConfig, dict]] = None,
        algorithm: str = "max",
        exclude: Sequence[str] = DEFAULT_EXCLUDE,
        amax_source: Optional[Union[str, Path, dict]] = None,
        strict: bool = True,
        white_list: Optional[list] = EMPTY_WHITE_LIST,
    ) -> None:
        super().__init__(white_list=white_list)
        self.rules = [rule if isinstance(rule, QuantRule) else QuantRule.from_dict(rule) for rule in rules or []]
        self.default = _coerce_module_config(default)
        if algorithm not in ("max",):
            raise NotImplementedError(
                f"calibration algorithm {algorithm!r} is not implemented; "
                "only 'max' (per-tensor max absolute value) is available"
            )
        self.algorithm = algorithm
        self.exclude = tuple(exclude)
        self.amax_source = amax_source
        self.strict = strict
        self._last_report: Optional[PlanReport] = None
        self._amax_map: Optional[AmaxMap] = None

    # ------------------------------------------------------------- NC contract
    def get_model_info(self, model, *args, **kwargs) -> list[Tuple[str, str]]:
        """Return ``(module_name, class_name)`` pairs for plan compilation."""
        return [(name, type(module).__name__) for name, module in model.named_modules() if name]

    def to_config_mapping(self, config_list=None, model_info: Optional[list] = None) -> "OrderedDict":
        """Compile rules into NC's ``{(op_name, op_type): ModuleQuantConfig}``."""
        report = compile_plan_from_info(
            model_info or [],
            self.rules,
            default=self.default,
            exclude=self.exclude,
            strict=self.strict,
        )
        self._last_report = report
        self._assign_amax(report)
        mapping = OrderedDict()
        for name, config in report.plan.items():
            mapping[(name, report.op_types[name])] = config
        return mapping

    def _assign_amax(self, report: PlanReport) -> None:
        """Fill per-module static ``amax`` from the external source before replacement."""
        amax_map = self.resolve_amax()
        if amax_map is None:
            return
        assigned, missing = 0, []
        for name, config in report.plan.items():
            fields = amax_map.lookup(name) or {}
            for field_name in ("weight", "input", "output"):
                quantizer_cfg = getattr(config, field_name)
                if quantizer_cfg is None or quantizer_cfg.scheme != "static":
                    continue
                if quantizer_cfg.amax is not None:
                    continue
                value = fields.get(field_name)
                if value is None:
                    missing.append(f"{name}.{field_name}")
                    continue
                quantizer_cfg.amax = value
                assigned += 1
        if missing and self.strict:
            head = ", ".join(missing[:5])
            raise ValueError(
                f"{len(missing)} static quantizers have no amax from {self.amax_source}: "
                f"{head} ...; fix the source or the name mapping"
            )
        logger.info(f"[amax] assigned {assigned} static amax values, missing={len(missing)}")

    # ------------------------------------------------------- tuning interface
    @classmethod
    def register_supported_configs(cls) -> list:
        """No tunable search space yet; fake-quant recipes are explicit."""
        return []

    @classmethod
    def get_config_set_for_tuning(cls) -> "FakeQuantConfig":
        """Return the default config used by autotune."""
        return cls.for_qat()

    # ---------------------------------------------------------------- helpers
    @property
    def last_report(self) -> Optional[PlanReport]:
        """Report from the most recent :meth:`to_config_mapping` call."""
        return self._last_report

    def needs_calibration(self) -> bool:
        """True when a static quantizer has no amax and no external source."""
        if self.amax_source is not None:
            return False
        configs = [self.default] if self.default is not None else []
        for rule in self.rules:
            configs.extend(cfg for cfg in (rule.weight, rule.input, rule.output) if cfg is not None)
        return any(cfg.scheme == "static" and cfg.amax is None for cfg in configs)

    def resolve_amax(self) -> Optional[AmaxMap]:
        """Load the configured amax source into an :class:`AmaxMap` (cached)."""
        if self.amax_source is None:
            return None
        if self._amax_map is not None:
            return self._amax_map
        if isinstance(self.amax_source, (str, Path)):
            path = Path(self.amax_source)
            translate = modelopt_megatron_translate if path.suffix != ".json" else None
        elif isinstance(self.amax_source, dict):
            path = Path(self.amax_source["file"])
            translate_name = self.amax_source.get("translate", "modelopt_megatron")
            if translate_name in (None, "none", "identity"):
                translate = None
            elif translate_name == "modelopt_megatron":
                translate = modelopt_megatron_translate
            else:
                raise ValueError(f"unknown amax translate policy {translate_name!r}")
        else:
            raise TypeError(f"unsupported amax_source {type(self.amax_source)!r}")
        if not path.is_file():
            raise FileNotFoundError(f"amax source not found: {path}")
        self._amax_map = AmaxMap(load_amax(path, translate=translate))
        return self._amax_map

    @classmethod
    def for_qat(cls, fmt: str = "mxfp4", *, weight: bool = True, input_quant: bool = True) -> "FakeQuantConfig":
        """Convenience preset replacing the legacy ``MXFP8`` QAT mapping."""
        return cls(
            rules=[
                QuantRule(
                    module=r".*",
                    weight=fmt if weight else None,
                    input=fmt if input_quant else None,
                )
            ]
        )

    # ------------------------------------------------------------ serialization
    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "rules": [rule.to_dict() for rule in self.rules],
            "default": self.default.to_dict() if self.default is not None else None,
            "algorithm": self.algorithm,
            "exclude": list(self.exclude),
            "amax_source": self.amax_source,
            "strict": self.strict,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "FakeQuantConfig":
        """Rebuild a config from :meth:`to_dict` output."""
        return cls(**config_dict)

    def __repr__(self) -> str:
        return (
            f"FakeQuantConfig(rules={len(self.rules)}, default={self.default}, "
            f"algorithm={self.algorithm!r}, amax_source={self.amax_source!r})"
        )


def print_last_summary(config: FakeQuantConfig, model=None) -> None:
    """Log the plan compiled by the most recent ``to_config_mapping``."""
    if config.last_report is None:
        logger.warning("no plan compiled yet; call to_config_mapping()/quantize() first")
        return
    from .plan import print_plan_summary

    print_plan_summary(config.last_report, model)


__all__ = ["FakeQuantConfig", "print_last_summary"]
