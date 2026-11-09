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
"""Per-quantizer options for fake quantization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Optional

import torch

from .formats import QuantFormat, get_format


def _default_scheme(fmt: QuantFormat) -> str:
    # MX/NV block formats are dynamic by construction; FP8 defaults to dynamic
    # too and only becomes static when the recipe asks for it explicitly.
    return "dynamic"


@dataclass
class QuantizerConfig:
    """Behaviour of one :class:`TensorQuantizer`.

    Attributes:
        format: Format name (``"mxfp4"``, ``"fp8_e4m3"``, ...) or a
            :class:`QuantFormat`.
        enabled: When False the quantizer is a no-op.
        scheme: ``"dynamic"`` (scale from the current tensor) or ``"static"``
            (scale from a calibrated ``amax``). ``None`` keeps the format default,
            which is ``"dynamic"`` for MX block formats.
        block_size: Optional override of the format's block size.
        calibrator: Calibration algorithm used for static quantizers.
        amax: Calibrated amax: a float/Tensor, ``"ptq"`` for an external
            summary file, or a mapping with a ``"file"`` key.
        fake_quant: When False the module is swapped but tensors are untouched
            (useful for debugging and export experiments).
    """

    format: QuantFormat | str = "bf16"
    enabled: bool = True
    scheme: Optional[str] = None
    block_size: Optional[int] = None
    calibrator: str = "max"
    amax: Any = None
    fake_quant: bool = True

    def __post_init__(self) -> None:
        """Coerce the format and validate the scheme."""
        self.format = get_format(self.format)
        if self.scheme is None:
            self.scheme = _default_scheme(self.format)
        if self.scheme not in ("dynamic", "static"):
            raise ValueError(f"scheme must be 'dynamic' or 'static', got {self.scheme!r}")
        if self.format.name == "bf16":
            self.enabled = False
        if self.scheme == "static" and self.format.name not in ("fp8_e4m3", "fp8_e5m2"):
            raise ValueError(f"static scheme is only implemented for per-tensor FP8, got {self.format.name}")
        if self.format.is_dynamic and self.scheme == "static":
            raise ValueError(f"{self.format.name} uses per-block scales; scheme='static' is not supported")
        if self.block_size is not None and self.block_size != self.format.block_size:
            raise ValueError(
                f"block_size override ({self.block_size}) does not match "
                f"{self.format.name} block size ({self.format.block_size})"
            )

    @property
    def is_noop(self) -> bool:
        """Return True when this config leaves tensors untouched."""
        return (not self.enabled) or (not self.fake_quant) or self.format.name == "bf16"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        payload: dict[str, Any] = {
            "format": self.format.name,
            "enabled": self.enabled,
            "scheme": self.scheme,
            "calibrator": self.calibrator,
            "fake_quant": self.fake_quant,
        }
        if self.block_size is not None:
            payload["block_size"] = self.block_size
        if isinstance(self.amax, torch.Tensor):
            payload["amax"] = float(self.amax.reshape(()))
        elif self.amax is not None:
            payload["amax"] = self.amax
        return payload

    @classmethod
    def from_any(cls, value: "QuantizerConfig | str | dict | None") -> Optional["QuantizerConfig"]:
        """Coerce ``str``/``dict``/``QuantizerConfig``/``None`` into a config."""
        if value is None or isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(format=value)
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError(f"cannot build QuantizerConfig from {type(value)!r}")


@dataclass
class ModuleQuantConfig:
    """Per-module policy: which quantizers exist and how each one behaves.

    Instances are the values of ``FakeQuantConfig.to_config_mapping()``.
    ``name`` is what NC's ``need_apply()`` compares against ``algos_mapping``
    keys, so the value object can travel through the existing dispatch path
    without subclassing :class:`TorchBaseConfig`.
    """

    weight: Optional[QuantizerConfig] = None
    input: Optional[QuantizerConfig] = None
    output: Optional[QuantizerConfig] = None

    name: ClassVar[str] = "fake_quant"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "weight": self.weight.to_dict() if self.weight is not None else None,
            "input": self.input.to_dict() if self.input is not None else None,
            "output": self.output.to_dict() if self.output is not None else None,
        }
