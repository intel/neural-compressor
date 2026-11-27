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
"""Fake-quantization configuration hierarchy (``FakeQuantConfig``, ``QuantSchemeRule``, ``QuantizerConfig``)."""

from __future__ import annotations

import copy
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Optional, Sequence, Tuple, Union

import torch

from neural_compressor.common.base_config import BaseConfig, register_config
from neural_compressor.common.utils import DEFAULT_WHITE_LIST, EMPTY_WHITE_LIST, FAKE_QUANT

from .formats import BF16, FP8_E4M3, QuantFormat, get_format
from .functional import QuantBackendName

ObserverName = Literal["minmax", "max"]

DEFAULT_IGNORE: tuple[str, ...] = (
    "lm_head",
    "re:.*embed.*",
)


class QuantizerConfig:
    """Configuration for a single tensor quantizer (weights, inputs, or outputs).

    Args:
        format: Format name (``"mxfp4"``, ``"nvfp4"``, ``"mxfp8"``, ``"fp8_e4m3"``,
            ``"bf16"``) or :class:`QuantFormat`.
        dynamic: Compute scale factors dynamically per forward tensor (``True``) or use a
            calibrated per-tensor ``amax`` buffer (``False``).
        enabled: Master switch; ``False`` makes the quantizer an identity pass-through.
        observer: Calibration observer for static quantization (``"max"`` or ``"minmax"``).
        backend: QDQ execution backend (``"auto"``, ``"pytorch"``, or ``"triton"``).
        amax: Optional pre-calibrated static maximum absolute value.
        scheme: Optional alias (``"dynamic"`` or ``"static"``) overriding ``dynamic``.
    """

    def __init__(
        self,
        format: Union[str, QuantFormat] = "mxfp4",
        dynamic: bool = True,
        enabled: bool = True,
        observer: ObserverName = "max",
        backend: QuantBackendName = "auto",
        amax: Optional[float] = None,
        *,
        scheme: Optional[Literal["dynamic", "static"]] = None,
    ) -> None:
        if isinstance(format, str) and format in ("fp8_static", "fp8_e4m3_static"):
            format = FP8_E4M3
            dynamic = False
        self.format: QuantFormat = get_format(format)
        if scheme is not None:
            if scheme not in ("dynamic", "static"):
                raise ValueError(f"scheme must be 'dynamic' or 'static', got {scheme!r}")
            dynamic = scheme == "dynamic"
        self.dynamic: bool = bool(dynamic)
        self.enabled: bool = bool(enabled) and self.format.name != "bf16"
        if observer not in ("minmax", "max"):
            raise ValueError(f"observer must be 'max' or 'minmax', got {observer!r}")
        self.observer: ObserverName = observer
        self.backend: QuantBackendName = backend
        self.amax: Optional[float] = float(amax) if amax is not None else None

        if self.enabled and not self.dynamic and not self.format.supports_static:
            raise ValueError(
                f"Format {self.format.name!r} supports only dynamic block-scaled quantization; set dynamic=True"
            )
        if self.amax is not None and self.amax < 0:
            raise ValueError(f"amax must be non-negative, got {self.amax}")

    @property
    def scheme(self) -> Literal["dynamic", "static"]:
        """Return ``'dynamic'`` or ``'static'``."""
        return "dynamic" if self.dynamic else "static"

    @property
    def is_noop(self) -> bool:
        """Return True when this quantizer performs no quantization."""
        return not self.enabled or self.format.name == "bf16"

    @classmethod
    def coerce(
        cls,
        spec: Union[None, str, QuantFormat, "QuantizerConfig", dict[str, Any]],
        *,
        default_backend: QuantBackendName = "auto",
    ) -> "QuantizerConfig":
        """Normalize shorthand strings, formats, dicts, or ``None`` into a :class:`QuantizerConfig`."""
        if spec is None:
            return cls(format=BF16, enabled=False, backend=default_backend)
        if isinstance(spec, QuantizerConfig):
            cloned = copy.copy(spec)
            if cloned.backend == "auto" and default_backend != "auto":
                cloned.backend = default_backend
            return cloned
        if isinstance(spec, QuantFormat):
            return cls(format=spec, dynamic=True, enabled=(spec.name != "bf16"), backend=default_backend)
        if isinstance(spec, str):
            norm = spec.strip().lower()
            if norm in ("none", "disabled", "bf16"):
                return cls(format=BF16, enabled=False, backend=default_backend)
            if norm in ("fp8_static", "fp8_e4m3_static"):
                return cls(format=FP8_E4M3, dynamic=False, enabled=True, backend=default_backend)
            return cls(format=get_format(norm), dynamic=True, enabled=True, backend=default_backend)
        if isinstance(spec, dict):
            return cls.from_dict(spec, default_backend=default_backend)
        raise TypeError(f"Cannot coerce {type(spec).__name__} into QuantizerConfig")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": self.format.name,
            "dynamic": self.dynamic,
            "enabled": self.enabled,
            "observer": self.observer,
            "backend": self.backend,
        }
        if self.amax is not None:
            payload["amax"] = self.amax
            payload["amax_hex"] = float(self.amax).hex()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, default_backend: QuantBackendName = "auto") -> "QuantizerConfig":
        amax_val: Optional[float] = None
        if "amax_hex" in payload and payload["amax_hex"] is not None:
            amax_val = float.fromhex(payload["amax_hex"])
        elif "amax" in payload and payload["amax"] is not None:
            amax_val = float(payload["amax"])
        return cls(
            format=payload.get("format", "mxfp4"),
            dynamic=payload.get("dynamic", True),
            enabled=payload.get("enabled", True),
            observer=payload.get("observer", "max"),
            backend=payload.get("backend", default_backend),
            amax=amax_val,
            scheme=payload.get("scheme"),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QuantizerConfig):
            return NotImplemented
        return (
            self.format == other.format
            and self.dynamic == other.dynamic
            and self.enabled == other.enabled
            and self.observer == other.observer
            and self.backend == other.backend
            and self.amax == other.amax
        )

    def __repr__(self) -> str:
        if self.is_noop:
            return "QuantizerConfig(disabled)"
        parts = [f"format={self.format.name!r}", f"dynamic={self.dynamic}"]
        if self.amax is not None:
            parts.append(f"amax={self.amax:.6g}")
        if self.backend != "auto":
            parts.append(f"backend={self.backend!r}")
        return f"QuantizerConfig({', '.join(parts)})"


QuantizerSpec = Union[None, str, QuantFormat, QuantizerConfig, dict[str, Any]]


@dataclass
class QuantSchemeRule:
    """Declarative rule mapping target modules to weight/activation quantizer specs.

    Target patterns follow ``compressed-tensors`` / ``ModelOpt`` conventions:
    * Class name: ``"Linear"`` matches supported linear module types.
    * Regex (prefixed with ``"re:"``): ``"re:.*mlp\\.(gate|up|down)_proj$"`` (full-string match).
    * Shell glob: ``"*mlp.experts*"`` (matched via :func:`fnmatch.fnmatchcase`).
    * Exact or suffix module name: ``"lm_head"`` matches ``"lm_head"`` or ``"*.lm_head"``.

    When multiple rules in :attr:`FakeQuantConfig.config_groups` match a module,
    later rules override earlier rules.
    """

    targets: list[str] = field(default_factory=lambda: ["Linear"])
    weights: QuantizerSpec = "mxfp4"
    input_activations: QuantizerSpec = "mxfp4"
    output_activations: QuantizerSpec = None

    def __post_init__(self) -> None:
        if isinstance(self.targets, str):
            self.targets = [self.targets]
        else:
            self.targets = list(self.targets)
        if not self.targets:
            raise ValueError("QuantSchemeRule.targets must contain at least one target pattern")

    def resolve(self, *, default_backend: QuantBackendName = "auto") -> "ModuleQuantConfig":
        """Materialize a concrete :class:`ModuleQuantConfig` from this rule."""
        return ModuleQuantConfig(
            weights=QuantizerConfig.coerce(self.weights, default_backend=default_backend),
            input_activations=QuantizerConfig.coerce(self.input_activations, default_backend=default_backend),
            output_activations=QuantizerConfig.coerce(self.output_activations, default_backend=default_backend),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "targets": list(self.targets),
            "weights": QuantizerConfig.coerce(self.weights).to_dict(),
            "input_activations": QuantizerConfig.coerce(self.input_activations).to_dict(),
            "output_activations": QuantizerConfig.coerce(self.output_activations).to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "QuantSchemeRule":
        return cls(
            targets=payload.get("targets", ["Linear"]),
            weights=payload.get("weights"),
            input_activations=payload.get("input_activations"),
            output_activations=payload.get("output_activations"),
        )


class ModuleQuantConfig:
    """Resolved per-module fake-quantization configuration (weights / inputs / outputs)."""

    name: str = FAKE_QUANT

    def __init__(
        self,
        weights: QuantizerSpec = None,
        input_activations: QuantizerSpec = None,
        output_activations: QuantizerSpec = None,
        *,
        weight: QuantizerSpec = None,
        input: QuantizerSpec = None,
        output: QuantizerSpec = None,
        default_backend: QuantBackendName = "auto",
    ) -> None:
        w_spec = weight if weights is None and weight is not None else weights
        i_spec = input if input_activations is None and input is not None else input_activations
        o_spec = output if output_activations is None and output is not None else output_activations
        self.weights: QuantizerConfig = QuantizerConfig.coerce(w_spec, default_backend=default_backend)
        self.input_activations: QuantizerConfig = QuantizerConfig.coerce(i_spec, default_backend=default_backend)
        self.output_activations: QuantizerConfig = QuantizerConfig.coerce(o_spec, default_backend=default_backend)

    @property
    def weight(self) -> QuantizerConfig:
        """Alias for ``weights``."""
        return self.weights

    @property
    def input(self) -> QuantizerConfig:
        """Alias for ``input_activations``."""
        return self.input_activations

    @property
    def output(self) -> QuantizerConfig:
        """Alias for ``output_activations``."""
        return self.output_activations

    @property
    def is_noop(self) -> bool:
        """Return True when weights, input_activations, and output_activations are all disabled."""
        return self.weights.is_noop and self.input_activations.is_noop and self.output_activations.is_noop

    def clone(self) -> "ModuleQuantConfig":
        return ModuleQuantConfig(
            weights=copy.deepcopy(self.weights),
            input_activations=copy.deepcopy(self.input_activations),
            output_activations=copy.deepcopy(self.output_activations),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": self.weights.to_dict(),
            "input_activations": self.input_activations.to_dict(),
            "output_activations": self.output_activations.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModuleQuantConfig":
        return cls(
            weights=payload.get("weights", payload.get("weight")),
            input_activations=payload.get("input_activations", payload.get("input")),
            output_activations=payload.get("output_activations", payload.get("output")),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModuleQuantConfig):
            return NotImplemented
        return (
            self.weights == other.weights
            and self.input_activations == other.input_activations
            and self.output_activations == other.output_activations
        )

    def __repr__(self) -> str:
        return (
            f"ModuleQuantConfig(weights={self.weights!r}, "
            f"input_activations={self.input_activations!r}, "
            f"output_activations={self.output_activations!r})"
        )


@register_config(framework_name="torch", algo_name=FAKE_QUANT)
class FakeQuantConfig(BaseConfig):
    """Top-level INC configuration for fake-quantization (QAT / QAD / PTQ simulation).

    Integrates directly with ``neural_compressor.torch.quantization.prepare``,
    ``quantize``, ``prepare_qat``, and ``convert``.

    Args:
        config_groups: Ordered list of :class:`QuantSchemeRule` entries. Defaults to
            MXFP4 dynamic weight + input quantization on all ``Linear`` modules.
        ignore: Module patterns excluded from quantization (checked before ``config_groups``).
            Defaults to ``["lm_head", "re:.*embed.*"]``.
        quantizer_state_path: Optional JSON file containing calibrated static ``amax`` values.
        backend: Default QDQ execution backend (``"auto"``, ``"pytorch"``, or ``"triton"``).
    """

    name = FAKE_QUANT
    params_list = ["config_groups", "ignore", "quantizer_state_path", "backend"]
    supported_configs: list = []

    def __init__(
        self,
        config_groups: Optional[Sequence[Union[QuantSchemeRule, dict[str, Any]]]] = None,
        ignore: Optional[Sequence[str]] = None,
        quantizer_state_path: Optional[Union[str, Path]] = None,
        backend: QuantBackendName = "auto",
        white_list: Optional[list[Union[str, Callable]]] = DEFAULT_WHITE_LIST,
    ) -> None:
        super().__init__(white_list=white_list)
        if config_groups is None:
            self.config_groups: list[QuantSchemeRule] = [
                QuantSchemeRule(targets=["Linear"], weights="mxfp4", input_activations="mxfp4")
            ]
        else:
            self.config_groups = [
                rule if isinstance(rule, QuantSchemeRule) else QuantSchemeRule.from_dict(rule) for rule in config_groups
            ]
        self.ignore: list[str] = list(DEFAULT_IGNORE) if ignore is None else list(ignore)
        self.quantizer_state_path: Optional[str] = (
            str(quantizer_state_path) if quantizer_state_path is not None else None
        )
        self.backend: QuantBackendName = backend
        self._post_init()

    # ------------------------------------------------------------------
    # Built-in Presets
    # ------------------------------------------------------------------

    @classmethod
    def mxfp4(
        cls,
        ignore: Optional[Sequence[str]] = None,
        *,
        backend: QuantBackendName = "auto",
    ) -> "FakeQuantConfig":
        """Dynamic W4A4 MXFP4 preset on all ``Linear`` layers."""
        return cls(
            config_groups=[QuantSchemeRule(targets=["Linear"], weights="mxfp4", input_activations="mxfp4")],
            ignore=ignore,
            backend=backend,
        )

    @classmethod
    def nvfp4(
        cls,
        ignore: Optional[Sequence[str]] = None,
        *,
        backend: QuantBackendName = "auto",
    ) -> "FakeQuantConfig":
        """Dynamic W4A4 NVFP4 preset on all ``Linear`` layers."""
        return cls(
            config_groups=[QuantSchemeRule(targets=["Linear"], weights="nvfp4", input_activations="nvfp4")],
            ignore=ignore,
            backend=backend,
        )

    @classmethod
    def mxfp8(
        cls,
        ignore: Optional[Sequence[str]] = None,
        *,
        backend: QuantBackendName = "auto",
    ) -> "FakeQuantConfig":
        """Dynamic W8A8 MXFP8 preset on all ``Linear`` layers."""
        return cls(
            config_groups=[QuantSchemeRule(targets=["Linear"], weights="mxfp8", input_activations="mxfp8")],
            ignore=ignore,
            backend=backend,
        )

    @classmethod
    def fp8_static(
        cls,
        ignore: Optional[Sequence[str]] = None,
        *,
        quantizer_state_path: Optional[Union[str, Path]] = None,
        backend: QuantBackendName = "auto",
    ) -> "FakeQuantConfig":
        """Calibrated static per-tensor FP8 E4M3 preset on all ``Linear`` layers."""
        return cls(
            config_groups=[QuantSchemeRule(targets=["Linear"], weights="fp8_static", input_activations="fp8_static")],
            ignore=ignore,
            quantizer_state_path=quantizer_state_path,
            backend=backend,
        )

    @classmethod
    def qwen3_5_mxfp4_hybrid(
        cls,
        quantizer_state_path: Optional[Union[str, Path]] = None,
        *,
        fp8_router: bool = False,
        backend: QuantBackendName = "auto",
    ) -> "FakeQuantConfig":
        """Hybrid MXFP4 + Static FP8 E4M3 recipe for Qwen3.5 dense & MoE Gated-DeltaNet models.

        * MLP / MoE routed & shared experts (``gate_proj``, ``up_proj``, ``down_proj``): MXFP4 dynamic.
        * Full self-attention (``q/k/v/o_proj``) & Gated DeltaNet main projections
          (``in_proj_qkv``, ``in_proj_z``, ``out_proj``): Static FP8 E4M3.
        * Sensitive DeltaNet decay/beta projections (``in_proj_a``, ``in_proj_b``), ``conv1d``,
          vision tower (``visual``), MTP modules, embeddings, and ``lm_head``: kept in BF16.
        * MoE routers (``mlp.gate``): BF16 by default, or static FP8 when ``fp8_router=True``.
        """
        ignore = [
            "lm_head",
            "re:.*embed.*",
            "re:.*visual.*",
            "re:.*mtp.*",
            r"re:.*linear_attn\.in_proj_(a|b|ba)$",
            "re:.*conv1d.*",
        ]
        if not fp8_router:
            ignore.append(r"re:.*mlp\.(gate|shared_expert_gate)$")
        else:
            ignore.append(r"re:.*mlp\.shared_expert_gate$")

        groups = [
            QuantSchemeRule(
                targets=[
                    r"re:.*mlp\.(experts|shared_expert).*(gate|up|down|gate_up)_proj$",
                    r"re:.*mlp\.(gate|up|down|gate_up)_proj$",
                ],
                weights="mxfp4",
                input_activations="mxfp4",
            ),
            QuantSchemeRule(
                targets=[
                    r"re:.*self_attn\.(q|k|v|o|qkv)_proj$",
                    r"re:.*linear_attn\.(in_proj_qkvz?|in_proj_z|out_proj)$",
                ],
                weights="fp8_static",
                input_activations="fp8_static",
            ),
        ]
        if fp8_router:
            groups.append(
                QuantSchemeRule(
                    targets=[r"re:.*mlp\.gate$"],
                    weights="fp8_static",
                    input_activations="fp8_static",
                )
            )
        return cls(
            config_groups=groups,
            ignore=ignore,
            quantizer_state_path=quantizer_state_path,
            backend=backend,
        )

    mxfp4_fp8_hybrid = qwen3_5_mxfp4_hybrid

    @classmethod
    def nvfp4_fp8_hybrid(
        cls,
        quantizer_state_path: Optional[Union[str, Path]] = None,
        *,
        fp8_router: bool = False,
        backend: QuantBackendName = "auto",
    ) -> "FakeQuantConfig":
        """Hybrid NVFP4 (W4A4 MLP/MoE) + Static FP8 E4M3 (Attention/DeltaNet) recipe."""
        cfg = cls.qwen3_5_mxfp4_hybrid(
            quantizer_state_path=quantizer_state_path,
            fp8_router=fp8_router,
            backend=backend,
        )
        cfg.config_groups[0] = QuantSchemeRule(
            targets=list(cfg.config_groups[0].targets),
            weights="nvfp4",
            input_activations="nvfp4",
        )
        return cfg

    # ------------------------------------------------------------------
    # INC BaseConfig Interface & Serialization
    # ------------------------------------------------------------------

    def get_params_dict(self) -> dict[str, Any]:
        return {param: getattr(self, param) for param in self.params_list}

    def needs_calibration(self) -> bool:
        """Return True if any rule specifies a static quantizer without a preloaded ``quantizer_state_path`` or ``amax``."""
        if self.quantizer_state_path is not None:
            return False
        for rule in self.config_groups:
            resolved = rule.resolve(default_backend=self.backend)
            for q_cfg in (resolved.weights, resolved.input_activations, resolved.output_activations):
                if not q_cfg.is_noop and not q_cfg.dynamic and q_cfg.amax is None:
                    return True
        return False

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "config_groups": [rule.to_dict() for rule in self.config_groups],
            "ignore": list(self.ignore),
            "backend": self.backend,
        }
        if self.quantizer_state_path is not None:
            payload["quantizer_state_path"] = self.quantizer_state_path
        return payload

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "FakeQuantConfig":
        return cls(
            config_groups=config_dict.get("config_groups", []),
            ignore=config_dict.get("ignore", list(DEFAULT_IGNORE)),
            quantizer_state_path=config_dict.get("quantizer_state_path"),
            backend=config_dict.get("backend", "auto"),
        )

    def to_json_file(self, json_file_path: Union[str, Path]) -> None:
        path = Path(json_file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({self.name: self.to_dict()}, indent=2, sort_keys=True) + "\n")

    @classmethod
    def from_json_file(cls, filename: Union[str, Path]) -> "FakeQuantConfig":
        raw = json.loads(Path(filename).read_text())
        if isinstance(raw, dict) and cls.name in raw and isinstance(raw[cls.name], dict):
            raw = raw[cls.name]
        return cls.from_dict(raw)

    @classmethod
    def from_yaml(cls, filename: Union[str, Path]) -> "FakeQuantConfig":
        """Load :class:`FakeQuantConfig` from a YAML file (either top-level or nested under ``quant:``)."""
        import yaml

        raw = yaml.safe_load(Path(filename).read_text(encoding="utf-8")) or {}
        if isinstance(raw, dict) and "quant" in raw and isinstance(raw["quant"], dict):
            raw = raw["quant"]
        elif isinstance(raw, dict) and cls.name in raw and isinstance(raw[cls.name], dict):
            raw = raw[cls.name]
        return cls.from_dict(raw)

    @classmethod
    def register_supported_configs(cls) -> list[Any]:
        from neural_compressor.torch.quantization.config import OperatorConfig

        cls.supported_configs = [
            OperatorConfig(config=cls, operators=EMPTY_WHITE_LIST, valid_func_list=[]),
        ]
        return cls.supported_configs

    @classmethod
    def get_config_set_for_tuning(cls) -> list["FakeQuantConfig"]:
        """Return default tuning candidates required by INC :class:`BaseConfig`."""
        return [cls.mxfp4(), cls.mxfp8()]

    @staticmethod
    def get_model_info(model: torch.nn.Module) -> list[Tuple[str, str]]:
        """Collect ``(module_name, op_type)`` pairs for all registered quantizable modules."""
        from .conversion import _collect_supported_modules

        return _collect_supported_modules(model)  # type: ignore[return-value]

    def to_config_mapping(
        self,
        config_list: Optional[list[BaseConfig]] = None,
        model_info: Optional[list[Tuple[str, Any]]] = None,
    ) -> OrderedDict[Tuple[str, Any], ModuleQuantConfig]:
        """Compile ``(module_name, op_type) -> ModuleQuantConfig`` mapping for INC ``quantize`` / ``prepare``."""
        from .conversion import match_quant_config_from_info

        config_list = [self] if config_list is None else config_list
        mapping: OrderedDict[Tuple[str, Any], ModuleQuantConfig] = OrderedDict()
        if not model_info:
            return mapping
        for cfg in config_list:
            if not isinstance(cfg, FakeQuantConfig):
                continue
            report = match_quant_config_from_info(model_info, cfg)
            for name, op_type in model_info:
                if name in report.plan:
                    mapping[(name, op_type)] = report.plan[name]
        return mapping


def get_default_fake_quant_config() -> FakeQuantConfig:
    """Return the default W4A4 MXFP4 :class:`FakeQuantConfig`."""
    return FakeQuantConfig.mxfp4()


MXFP4_DEFAULT_CFG = FakeQuantConfig.mxfp4()
NVFP4_DEFAULT_CFG = FakeQuantConfig.nvfp4()
MXFP8_DEFAULT_CFG = FakeQuantConfig.mxfp8()
FP8_DEFAULT_CFG = FakeQuantConfig.fp8_static()
MXFP4_FP8_HYBRID_CFG = FakeQuantConfig.mxfp4_fp8_hybrid()
NVFP4_FP8_HYBRID_CFG = FakeQuantConfig.nvfp4_fp8_hybrid()


__all__ = [
    "DEFAULT_IGNORE",
    "FP8_DEFAULT_CFG",
    "FakeQuantConfig",
    "MXFP4_DEFAULT_CFG",
    "MXFP4_FP8_HYBRID_CFG",
    "MXFP8_DEFAULT_CFG",
    "ModuleQuantConfig",
    "NVFP4_DEFAULT_CFG",
    "NVFP4_FP8_HYBRID_CFG",
    "ObserverName",
    "QuantSchemeRule",
    "QuantizerConfig",
    "QuantizerSpec",
    "get_default_fake_quant_config",
]
