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
"""Quantization format specs shared by QAT and QAD.

A format is a first-class object instead of ModelOpt's ``num_bits=(E, M)`` tuple:
recipes and code both speak ``"mxfp4"`` / ``"fp8_e4m3"``, and every property the
quantizer needs (block size, scale kind, element range) is explicit.
"""

from __future__ import annotations

from dataclasses import dataclass

E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
E4M3_MAX = 448.0
E2M1_MAX = 6.0


@dataclass(frozen=True)
class QuantFormat:
    """Static description of one quantization format.

    Attributes:
        name: Registry key, e.g. ``"mxfp4"``.
        element_bits: Bits per element (4, 8 or 16).
        block_size: Elements sharing one scale factor; ``None`` means per-tensor.
        scale: Scale representation: ``"e8m0"``, ``"e4m3"``, ``"fp32"`` or ``None``.
        exponent_bits: Exponent bits for FP formats (E2M1 -> 2).
        mantissa_bits: Mantissa bits for FP formats (E2M1 -> 1).
        global_scale: Whether a second-level global scale is used (NVFP4).
        max_value: Largest representable magnitude.
        signed: Whether negative values are representable.
    """

    name: str
    element_bits: int
    block_size: int | None = None
    scale: str | None = None
    exponent_bits: int | None = None
    mantissa_bits: int | None = None
    global_scale: bool = False
    max_value: float | None = None
    signed: bool = True

    @property
    def is_float(self) -> bool:
        """Return True for floating-point style formats."""
        return self.exponent_bits is not None

    @property
    def is_dynamic(self) -> bool:
        """Return True when the scale is computed from the current tensor."""
        return self.scale in ("e8m0", "e4m3") and self.block_size is not None

    def __str__(self) -> str:
        return self.name


MXFP4 = QuantFormat(
    name="mxfp4",
    element_bits=4,
    block_size=32,
    scale="e8m0",
    exponent_bits=2,
    mantissa_bits=1,
    max_value=E2M1_MAX,
)
NVFP4 = QuantFormat(
    name="nvfp4",
    element_bits=4,
    block_size=16,
    scale="e4m3",
    exponent_bits=2,
    mantissa_bits=1,
    global_scale=True,
    max_value=E2M1_MAX,
)
MXFP8 = QuantFormat(
    name="mxfp8",
    element_bits=8,
    block_size=32,
    scale="e8m0",
    exponent_bits=4,
    mantissa_bits=3,
    max_value=E4M3_MAX,
)
FP8_E4M3 = QuantFormat(
    name="fp8_e4m3",
    element_bits=8,
    block_size=None,
    scale="fp32",
    exponent_bits=4,
    mantissa_bits=3,
    max_value=E4M3_MAX,
)
BF16 = QuantFormat(name="bf16", element_bits=16)

_FORMATS = {fmt.name: fmt for fmt in (MXFP4, MXFP8, NVFP4, FP8_E4M3, BF16)}


def register_format(fmt: QuantFormat) -> QuantFormat:
    """Register a custom format (used by tests and downstream extensions)."""
    if not isinstance(fmt, QuantFormat):
        raise TypeError(f"expected QuantFormat, got {type(fmt)!r}")
    _FORMATS[fmt.name] = fmt
    return fmt


def get_format(fmt: str | QuantFormat) -> QuantFormat:
    """Resolve a format name or pass a :class:`QuantFormat` through."""
    if isinstance(fmt, QuantFormat):
        return fmt
    try:
        return _FORMATS[fmt]
    except KeyError as exc:
        raise ValueError(f"unknown quantization format {fmt!r}; known: {sorted(_FORMATS)}") from exc


def list_formats() -> list[str]:
    """Return the registered format names."""
    return sorted(_FORMATS)
