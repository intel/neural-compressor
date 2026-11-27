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
"""Cold-path weight bit-packing and dequantization utilities for checkpoint export.

This module is strictly separated from :mod:`.functional` so nibble packing/unpacking
never executes during QAT/QAD training forward or activation recomputation.
Packed layouts follow ``compressed-tensors`` and ``vLLM`` conventions:

* MXFP4: ``data`` uint8 ``[out, in // 2]`` (low nibble = even column),
  ``scale`` uint8 E8M0 ``[out, in // 32]`` (``exponent + 127``).
* NVFP4: ``data`` uint8 ``[out, in // 2]``, ``scale`` float8_e4m3fn ``[out, in // 16]``,
  ``global_scale`` float32 ``[1]``.
* MXFP8: ``data`` float8_e4m3fn ``[out, in]``, ``scale`` uint8 E8M0 ``[out, in // 32]``.
* Static FP8 E4M3: ``data`` float8_e4m3fn ``[out, in]``, ``scale`` float32 ``[1]``
  (computed via exact dual-FP32 reciprocal ``1 / (448 / amax)`` so dequantization
  is 0-ULP identical to training QDQ).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .formats import (
    BF16,
    E2M1_MAX,
    E4M3_MAX,
    FP8_E4M3,
    MXFP4,
    MXFP8,
    NVFP4,
    QuantFormat,
    get_format,
)
from .functional import _check_block_dim, _get_e2m1_luts


@dataclass
class PackedWeightTensor:
    """Packed on-disk representation of a quantized weight tensor."""

    data: torch.Tensor
    scale: Optional[torch.Tensor]
    global_scale: Optional[torch.Tensor]
    original_shape: tuple[int, ...]
    format: QuantFormat

    @property
    def nbytes(self) -> int:
        """Total compressed footprint in bytes (data + scales)."""
        total = self.data.numel() * self.data.element_size()
        if self.scale is not None:
            total += self.scale.numel() * self.scale.element_size()
        if self.global_scale is not None:
            total += self.global_scale.numel() * self.global_scale.element_size()
        return total


def _e2m1_code(x: torch.Tensor) -> torch.Tensor:
    """Encode scaled FP32 values into 4-bit E2M1 codes (bit 3 = sign, bits 0..2 = magnitude)."""
    bounds, _ = _get_e2m1_luts(x.device)
    absx = x.abs()
    order = torch.searchsorted(bounds, absx).to(torch.uint8)
    tie = ((absx == 0.75) | (absx == 1.75) | (absx == 3.5)).to(torch.uint8)
    sign = (x < 0).to(torch.uint8)
    return (sign << 3) + order + tie


def pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """Pack pairs of 4-bit codes per byte (even column in low nibble, odd column in high nibble)."""
    if codes.shape[-1] % 2 != 0:
        raise ValueError(f"Last dimension must be even to pack 4-bit codes, got {codes.shape[-1]}")
    lo = codes[..., 0::2] & 0x0F
    hi = codes[..., 1::2] & 0x0F
    return ((hi << 4) | lo).to(torch.uint8)


def unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 bytes back into 4-bit codes (inverse of :func:`pack_nibbles`)."""
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    out = torch.empty((*packed.shape[:-1], packed.shape[-1] * 2), dtype=torch.uint8, device=packed.device)
    out[..., 0::2] = lo
    out[..., 1::2] = hi
    return out


def _codes_to_fp32(codes: torch.Tensor) -> torch.Tensor:
    _, values = _get_e2m1_luts(codes.device)
    mag = values[(codes & 0x07).long()]
    return torch.where((codes & 0x08) != 0, -mag, mag)


def _pack_mxfp4(w: torch.Tensor) -> PackedWeightTensor:
    _check_block_dim(w, 32)
    orig_shape = tuple(w.shape)
    lead_shape = orig_shape[:-1]
    k = orig_shape[-1]
    blocks = w.float().reshape(-1, k // 32, 32)
    amax = blocks.abs().amax(dim=-1)
    descale = amax / E2M1_MAX
    e8m0 = torch.ceil(torch.maximum(torch.log2(descale), torch.full_like(descale, -127.0)))
    e8m0 = torch.where(amax > 0, e8m0, torch.full_like(e8m0, -127.0))
    encoded_scale = (e8m0 + 127.0).clamp(0.0, 255.0).to(torch.uint8)
    scale_f32 = torch.exp2(encoded_scale.float() - 127.0).unsqueeze(-1)
    codes = _e2m1_code(blocks / scale_f32).reshape(*lead_shape, k)
    return PackedWeightTensor(
        data=pack_nibbles(codes).contiguous(),
        scale=encoded_scale.reshape(*lead_shape, k // 32).contiguous(),
        global_scale=None,
        original_shape=orig_shape,
        format=MXFP4,
    )


def _pack_nvfp4(w: torch.Tensor, amax: Optional[torch.Tensor] = None) -> PackedWeightTensor:
    _check_block_dim(w, 16)
    orig_shape = tuple(w.shape)
    lead_shape = orig_shape[:-1]
    k = orig_shape[-1]
    wf = w.float()
    global_amax = wf.abs().amax() if amax is None else amax.to(device=w.device, dtype=torch.float32).reshape(())
    raw_global = global_amax / (E2M1_MAX * E4M3_MAX)
    global_scale = torch.where(raw_global > 0, raw_global, torch.ones_like(raw_global))
    blocks = wf.reshape(-1, k // 16, 16)
    block_amax = blocks.abs().amax(dim=-1)
    block_scale_fp8 = (block_amax / (E2M1_MAX * global_scale)).clamp(2.0**-9, E4M3_MAX).to(torch.float8_e4m3fn)
    eff_scale = (block_scale_fp8.float() * global_scale).unsqueeze(-1)
    codes = _e2m1_code(blocks / eff_scale).reshape(*lead_shape, k)
    return PackedWeightTensor(
        data=pack_nibbles(codes).contiguous(),
        scale=block_scale_fp8.reshape(*lead_shape, k // 16).contiguous(),
        global_scale=global_scale.reshape(1).contiguous(),
        original_shape=orig_shape,
        format=NVFP4,
    )


def _pack_mxfp8(w: torch.Tensor) -> PackedWeightTensor:
    _check_block_dim(w, 32)
    orig_shape = tuple(w.shape)
    lead_shape = orig_shape[:-1]
    k = orig_shape[-1]
    blocks = w.float().reshape(-1, k // 32, 32)
    amax = blocks.abs().amax(dim=-1)
    descale = amax / E4M3_MAX
    e8m0 = torch.ceil(torch.maximum(torch.log2(descale), torch.full_like(descale, -127.0)))
    e8m0 = torch.where(amax > 0, e8m0, torch.full_like(e8m0, -127.0))
    encoded_scale = (e8m0 + 127.0).clamp(0.0, 255.0).to(torch.uint8)
    scale_f32 = torch.exp2(encoded_scale.float() - 127.0).unsqueeze(-1)
    q_data = (blocks / scale_f32).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return PackedWeightTensor(
        data=q_data.reshape(orig_shape).contiguous(),
        scale=encoded_scale.reshape(*lead_shape, k // 32).contiguous(),
        global_scale=None,
        original_shape=orig_shape,
        format=MXFP8,
    )


def _pack_fp8_e4m3(w: torch.Tensor, amax: Optional[torch.Tensor] = None) -> PackedWeightTensor:
    orig_shape = tuple(w.shape)
    wf = w.float()
    if amax is not None:
        amax_f32 = amax.to(device=w.device, dtype=torch.float32).reshape(())
        safe_amax = torch.where(amax_f32 <= 2.0**-24, torch.ones_like(amax_f32), amax_f32)
        qscale = torch.full_like(safe_amax, E4M3_MAX) / safe_amax
        inv_scale = torch.full_like(qscale, 1.0) / qscale
        q_data = (wf * qscale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        return PackedWeightTensor(
            data=q_data.contiguous(),
            scale=inv_scale.reshape(1).contiguous(),
            global_scale=None,
            original_shape=orig_shape,
            format=FP8_E4M3,
        )
    dyn_amax = wf.abs().amax()
    raw_scale = dyn_amax / E4M3_MAX
    safe_scale = torch.where(raw_scale > 0, raw_scale, torch.ones_like(raw_scale))
    q_data = (wf / safe_scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return PackedWeightTensor(
        data=q_data.contiguous(),
        scale=safe_scale.reshape(1).contiguous(),
        global_scale=None,
        original_shape=orig_shape,
        format=FP8_E4M3,
    )


@torch.no_grad()
def pack_weight(
    weight: torch.Tensor,
    fmt: str | QuantFormat = "mxfp4",
    *,
    amax: Optional[torch.Tensor] = None,
) -> PackedWeightTensor:
    """Pack ``weight`` into the compressed on-disk layout of ``fmt``."""
    spec = get_format(fmt)
    if spec.name == "bf16":
        return PackedWeightTensor(
            data=weight.contiguous(),
            scale=None,
            global_scale=None,
            original_shape=tuple(weight.shape),
            format=BF16,
        )
    if spec.name == "mxfp4":
        return _pack_mxfp4(weight)
    if spec.name == "nvfp4":
        return _pack_nvfp4(weight, amax=amax)
    if spec.name == "mxfp8":
        return _pack_mxfp8(weight)
    if spec.name == "fp8_e4m3":
        return _pack_fp8_e4m3(weight, amax=amax)
    raise ValueError(f"Unsupported format {spec.name!r} for weight packing")


@torch.no_grad()
def dequantize_packed_weight(
    packed: PackedWeightTensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize a :class:`PackedWeightTensor` back to dense floating-point."""
    name = packed.format.name
    if name == "bf16":
        return packed.data.to(dtype)
    if name == "mxfp4":
        assert packed.scale is not None
        codes = unpack_nibbles(packed.data).reshape(-1, 32)
        vals = _codes_to_fp32(codes)
        scales = torch.exp2(packed.scale.float().reshape(-1, 1) - 127.0)
        return (vals * scales).reshape(packed.original_shape).to(dtype)
    if name == "nvfp4":
        assert packed.scale is not None and packed.global_scale is not None
        codes = unpack_nibbles(packed.data).reshape(-1, 16)
        vals = _codes_to_fp32(codes)
        scales = packed.scale.float().reshape(-1, 1) * packed.global_scale.float().reshape(1, 1)
        return (vals * scales).reshape(packed.original_shape).to(dtype)
    if name == "mxfp8":
        assert packed.scale is not None
        vals = packed.data.float().reshape(-1, 32)
        scales = torch.exp2(packed.scale.float().reshape(-1, 1) - 127.0)
        return (vals * scales).reshape(packed.original_shape).to(dtype)
    if name == "fp8_e4m3":
        assert packed.scale is not None
        return (packed.data.float() * packed.scale.float().reshape(())).reshape(packed.original_shape).to(dtype)
    raise ValueError(f"Unsupported packed format {name!r}")


__all__ = [
    "PackedWeightTensor",
    "dequantize_packed_weight",
    "pack_nibbles",
    "pack_weight",
    "unpack_nibbles",
]
