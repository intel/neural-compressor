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
"""TensorQuantizer: the single building block behind QAT and QAD."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .formats import E4M3_MAX, QuantFormat, get_format
from .ops.reference import (
    _e2m1_code,
    dequantize,
    fake_quant,
    fake_quant_fp8_amax,
    fake_quant_fp8_static,
    pack_nibbles,
    quantize,
)
from .options import QuantizerConfig


class TensorQuantizer(nn.Module):
    """Quantize-dequantize one tensor with a straight-through gradient.

    Dynamic formats (MXFP4/NVFP4) compute their block scales on every forward.
    Static formats (per-tensor FP8) consume a calibrated ``amax`` buffer, which
    is persistent so a checkpoint can rebuild the quantizer exactly.

    Args:
        config: Quantizer behaviour. ``amax`` may carry a calibrated value;
            otherwise :meth:`load_amax` or calibration must provide one.
    """

    def __init__(self, config: QuantizerConfig | str | dict) -> None:
        super().__init__()
        self.config = QuantizerConfig.from_any(config)
        self.format: QuantFormat = get_format(self.config.format)
        self.scheme = self.config.scheme
        self.enabled = (not self.config.is_noop) and self.format.name != "bf16"
        self.fake_quant = self.config.fake_quant
        self.calibrating = False

        amax = self.config.amax
        if isinstance(amax, (int, float)):
            amax = torch.tensor(float(amax), dtype=torch.float32)
        elif isinstance(amax, torch.Tensor):
            amax = amax.detach().reshape(()).to(torch.float32)
        elif amax is not None:
            # "ptq" / {"file": ...} are resolved by the calibration source, not here.
            amax = None
        self.register_buffer("amax", amax, persistent=True)

    # ------------------------------------------------------------------ state
    def load_amax(self, amax: torch.Tensor | float) -> None:
        """Set the calibrated amax (fp32, scalar)."""
        value = amax if isinstance(amax, torch.Tensor) else torch.tensor(float(amax))
        self.amax = value.detach().reshape(()).to(torch.float32)

    def finalize(self) -> None:
        """Leave calibration mode; static quantizers now require an amax."""
        self.calibrating = False
        if self.scheme == "static" and self.enabled and self.amax is None:
            raise RuntimeError(
                f"{self._get_name()} is static but has no amax; run calibration or load it from a summary"
            )

    def _update_amax(self, x: torch.Tensor) -> None:
        current = x.detach().float().abs().amax()
        if self.amax is None:
            self.amax = current.reshape(()).to(torch.float32)
        else:
            self.amax = torch.maximum(self.amax, current).reshape(()).to(torch.float32)

    # ---------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return fake-quantized ``x`` with an identity backward."""
        if not self.enabled:
            return x
        if self.scheme == "static":
            if self.calibrating:
                self._update_amax(x)
            if self.amax is None:
                raise RuntimeError(
                    f"static quantizer for {self.format.name} has no amax; "
                    "call calibrate()/load_amax() before the forward pass"
                )
            quantized = fake_quant_fp8_amax(x, self.amax)
        else:
            quantized = fake_quant(x, self.format)
        return x + (quantized - x).detach()

    def _fake_quantize(self, inputs: torch.Tensor):
        """Return QDQ weights and the E8M0 exponent used by legacy exporters."""
        if self.format.name not in ("mxfp4", "mxfp8"):
            raise NotImplementedError(f"legacy packing does not support {self.format.name}")
        quantized = quantize(inputs, self.format)
        return dequantize(quantized).to(inputs.dtype), quantized.scales.float() - 127.0

    def weight_pack(self, weight: torch.Tensor, scale: torch.Tensor):
        """Pack QDQ weights using the exponent returned by ``_fake_quantize``."""
        if self.format.name not in ("mxfp4", "mxfp8"):
            raise NotImplementedError(f"weight_pack does not support {self.format.name}")
        original_shape = weight.shape
        rows = weight.reshape(-1, original_shape[-1])
        scaled = rows / torch.exp2(scale.float()).repeat_interleave(32, dim=-1)
        encoded_scale = (scale + 127).to(torch.uint8)
        if self.format.name == "mxfp4":
            packed = pack_nibbles(_e2m1_code(scaled)).reshape(original_shape[0], original_shape[1] // 2)
        else:
            packed = scaled.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).reshape(original_shape)
        return packed, encoded_scale.reshape(original_shape[0], -1)

    # ------------------------------------------------- legacy attribute views
    @property
    def num_bits(self) -> int:
        """Element bits, kept for the legacy QAT/export helpers."""
        return self.format.element_bits

    @property
    def data_type(self) -> str:
        """Legacy auto_round-style data type name."""
        return {
            "mxfp8": "mx_fp8",
            "mxfp4": "mx_fp4",
            "fp8_e4m3": "fp8_e4m3",
        }.get(self.format.name, self.format.name)

    @property
    def _disabled(self) -> bool:
        """Legacy ``_disabled`` flag."""
        return not self.enabled

    def extra_repr(self) -> str:
        """Human-readable state used by plan summaries and ``print(model)``."""
        amax = "none" if self.amax is None else f"{float(self.amax):.4g}"
        return f"format={self.format.name}, scheme={self.scheme}, " f"enabled={self.enabled}, amax={amax}"


# Backwards-compatible alias for the legacy QAT path.
def build_quantizer(config: QuantizerConfig | str | dict) -> TensorQuantizer:
    """Factory kept separate so call sites read ``build_quantizer(cfg)``."""
    return TensorQuantizer(config)


__all__ = ["TensorQuantizer", "build_quantizer", "fake_quant_fp8_static"]
