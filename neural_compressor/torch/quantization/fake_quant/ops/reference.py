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
"""Pure-torch reference kernels for fake quantization.

Numerics follow ModelOpt's MXFP4QTensor / NVFP4QTensor and TransformerEngine's
reference implementation:

* MXFP4: E2M1 elements, block 32, E8M0 scale = 2^ceil(log2(amax / 6)).
* NVFP4: E2M1 elements, block 16, E4M3 block scale + FP32 global scale.
* FP8: E4M3, per-tensor scale from a calibrated amax.

The packed nibble order is low nibble = even column, high nibble = odd column,
matching ModelOpt / compressed-tensors / vLLM. This module is the numeric
ground truth every optional backend (Triton, vendor kernels) is checked against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..formats import (
    BF16,
    E2M1_BOUNDS,
    E2M1_MAX,
    E2M1_VALUES,
    E4M3_MAX,
    FP8_E4M3,
    MXFP4,
    MXFP8,
    NVFP4,
    QuantFormat,
    get_format,
)

_E2M1_VALUES = torch.tensor(E2M1_VALUES, dtype=torch.float32)
_E2M1_BOUNDS = torch.tensor(E2M1_BOUNDS, dtype=torch.float32)


@dataclass
class QuantizedTensor:
    """Packed result of :func:`quantize`."""

    data: torch.Tensor
    scales: Optional[torch.Tensor]
    global_scale: Optional[torch.Tensor]
    shape: tuple[int, ...]
    fmt: QuantFormat

    @property
    def nbytes(self) -> int:
        """Compressed size in bytes (data + scales)."""
        nbytes = self.data.numel() * self.data.element_size()
        if self.scales is not None:
            nbytes += self.scales.numel() * self.scales.element_size()
        if self.global_scale is not None:
            nbytes += self.global_scale.numel() * self.global_scale.element_size()
        return nbytes


def _check_last_dim(x: torch.Tensor, block: int) -> int:
    if x.ndim < 2:
        raise ValueError(f"quantize expects at least 2-D input, got {tuple(x.shape)}")
    k = x.shape[-1]
    if k % block:
        raise ValueError(f"last dim {k} is not a multiple of block {block}")
    return k


def _rows(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, x.shape[-1])


def _unflatten_rows(rows: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    return rows.reshape(shape)


def _e2m1_code(x: torch.Tensor) -> torch.Tensor:
    # Round-half-to-even: the only ties that round up are the odd-code bounds
    # [0.75, 1.75, 3.5]. Same rule as ModelOpt's CUDA MXFP4/NVFP4 quantizers.
    bounds = _E2M1_BOUNDS.to(x.device)
    absx = x.abs()
    order = torch.searchsorted(bounds, absx).to(torch.uint8)
    odd = bounds[[1, 3, 5]]
    tie = (absx.unsqueeze(-1) == odd).any(dim=-1).to(torch.uint8)
    sign = (x < 0).to(torch.uint8)
    return (sign << 3) + order + tie


def pack_nibbles(code: torch.Tensor) -> torch.Tensor:
    """Pack two 4-bit codes per byte (low nibble first)."""
    if code.shape[-1] % 2:
        raise ValueError(f"last dim must be even to pack FP4, got {code.shape[-1]}")
    lo = code[..., 0::2]
    hi = code[..., 1::2]
    return ((hi << 4) | lo).to(torch.uint8)


def unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`pack_nibbles`."""
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    out = torch.empty(
        (*packed.shape[:-1], packed.shape[-1] * 2),
        dtype=torch.uint8,
        device=packed.device,
    )
    out[..., 0::2] = lo
    out[..., 1::2] = hi
    return out


def _codes_to_values(code: torch.Tensor, scale: Optional[torch.Tensor] = None) -> torch.Tensor:
    lut = _E2M1_VALUES.to(code.device)
    magnitude = lut[(code & 0x07).long()]
    sign = torch.where((code & 0x08) > 0, -1.0, 1.0)
    out = magnitude * sign
    if scale is not None:
        out = out * scale
    return out


def _mxfp4_quantize(x: torch.Tensor) -> QuantizedTensor:
    shape = tuple(x.shape)
    rows = _rows(x.float())
    data_parts, scale_parts = [], []
    chunk = 2048
    for start in range(0, rows.shape[0], chunk):
        block_rows = rows[start : start + chunk]
        blocks = block_rows.view(block_rows.shape[0], -1, 32)
        amax = blocks.abs().amax(dim=-1)
        descale = amax / E2M1_MAX
        e8m0 = torch.ceil(torch.maximum(torch.log2(descale), torch.full_like(descale, -127.0)))
        e8m0 = torch.where(amax > 0, e8m0, torch.full_like(e8m0, -127.0))
        scaled = blocks / torch.exp2(e8m0).unsqueeze(-1)
        code = _e2m1_code(scaled).reshape(block_rows.shape[0], -1)
        data_parts.append(pack_nibbles(code))
        scale_parts.append((e8m0 + 127).clamp(0, 255).to(torch.uint8))
    return QuantizedTensor(
        data=torch.cat(data_parts, dim=0),
        scales=torch.cat(scale_parts, dim=0),
        global_scale=None,
        shape=shape,
        fmt=MXFP4,
    )


def _nvfp4_quantize(x: torch.Tensor) -> QuantizedTensor:
    shape = tuple(x.shape)
    rows = _rows(x.float())
    global_amax = rows.abs().amax()
    global_scale = global_amax / (E2M1_MAX * E4M3_MAX)
    if float(global_scale) == 0.0:
        global_scale = torch.ones_like(global_scale)
    data_parts, scale_parts = [], []
    chunk = 2048
    for start in range(0, rows.shape[0], chunk):
        block_rows = rows[start : start + chunk]
        blocks = block_rows.view(block_rows.shape[0], -1, 16)
        amax = blocks.abs().amax(dim=-1)
        block_scale = (amax / (E2M1_MAX * global_scale)).clamp(2**-9, E4M3_MAX)
        block_scale = block_scale.to(torch.float8_e4m3fn).float()
        scaled = blocks / (block_scale * global_scale).unsqueeze(-1)
        code = _e2m1_code(scaled).reshape(block_rows.shape[0], -1)
        data_parts.append(pack_nibbles(code))
        scale_parts.append(block_scale.to(torch.float8_e4m3fn))
    return QuantizedTensor(
        data=torch.cat(data_parts, dim=0),
        scales=torch.cat(scale_parts, dim=0),
        global_scale=global_scale.reshape(()),
        shape=shape,
        fmt=NVFP4,
    )


def _mxfp8_quantize(x: torch.Tensor) -> QuantizedTensor:
    """E4M3 elements with a per-32-block E8M0 scale (ModelOpt's MXFP8)."""
    shape = tuple(x.shape)
    rows = _rows(x.float())
    data_parts, scale_parts = [], []
    chunk = 2048
    for start in range(0, rows.shape[0], chunk):
        block_rows = rows[start : start + chunk]
        blocks = block_rows.view(block_rows.shape[0], -1, 32)
        amax = blocks.abs().amax(dim=-1)
        descale = amax / E4M3_MAX
        e8m0 = torch.ceil(torch.maximum(torch.log2(descale), torch.full_like(descale, -127.0)))
        e8m0 = torch.where(amax > 0, e8m0, torch.full_like(e8m0, -127.0))
        scaled = (blocks / torch.exp2(e8m0).unsqueeze(-1)).clamp(-E4M3_MAX, E4M3_MAX)
        data_parts.append(scaled.to(torch.float8_e4m3fn).reshape(block_rows.shape[0], -1))
        scale_parts.append((e8m0 + 127).clamp(0, 255).to(torch.uint8))
    return QuantizedTensor(
        data=torch.cat(data_parts, dim=0),
        scales=torch.cat(scale_parts, dim=0),
        global_scale=None,
        shape=shape,
        fmt=MXFP8,
    )


def _fp8_quantize(x: torch.Tensor) -> QuantizedTensor:
    shape = tuple(x.shape)
    xf = x.float()
    amax = xf.abs().amax()
    scale = amax / E4M3_MAX
    if float(scale) == 0.0:
        scale = torch.ones_like(scale)
    quantized = (xf / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return QuantizedTensor(
        data=quantized,
        scales=scale.reshape(()),
        global_scale=None,
        shape=shape,
        fmt=FP8_E4M3,
    )


def quantize(x: torch.Tensor, fmt: str | QuantFormat = "mxfp4") -> QuantizedTensor:
    """Quantize ``x`` into the packed representation of ``fmt``."""
    spec = get_format(fmt)
    if spec.name == "bf16":
        return QuantizedTensor(data=x, scales=None, global_scale=None, shape=tuple(x.shape), fmt=BF16)
    with torch.no_grad():
        if spec.name == "mxfp4":
            return _mxfp4_quantize(x)
        if spec.name == "mxfp8":
            return _mxfp8_quantize(x)
        if spec.name == "nvfp4":
            return _nvfp4_quantize(x)
        if spec.name == "fp8_e4m3":
            return _fp8_quantize(x)
    raise ValueError(f"unsupported format {spec.name}")


def _mxfp4_dequantize(qt: QuantizedTensor) -> torch.Tensor:
    code = unpack_nibbles(qt.data)[:, : qt.shape[-1]]
    scale = torch.exp2(qt.scales.float() - 127.0).repeat_interleave(32, dim=-1)
    out = _codes_to_values(code, scale)
    return _unflatten_rows(out, qt.shape)


def _nvfp4_dequantize(qt: QuantizedTensor) -> torch.Tensor:
    code = unpack_nibbles(qt.data)[:, : qt.shape[-1]]
    block_scale = qt.scales.float().repeat_interleave(16, dim=-1)
    out = _codes_to_values(code, block_scale * qt.global_scale.float())
    return _unflatten_rows(out, qt.shape)


def _mxfp8_dequantize(qt: QuantizedTensor) -> torch.Tensor:
    scale = torch.exp2(qt.scales.float() - 127.0).repeat_interleave(32, dim=-1)
    out = qt.data.float() * scale.reshape(qt.data.shape)
    return _unflatten_rows(out, qt.shape)


def _fp8_dequantize(qt: QuantizedTensor) -> torch.Tensor:
    return qt.data.float() * qt.scales.float()


def dequantize(qt: QuantizedTensor) -> torch.Tensor:
    """Expand a :class:`QuantizedTensor` back to dense float values."""
    if qt.fmt.name == "bf16":
        return qt.data
    if qt.fmt.name == "mxfp4":
        return _mxfp4_dequantize(qt)
    if qt.fmt.name == "mxfp8":
        return _mxfp8_dequantize(qt)
    if qt.fmt.name == "nvfp4":
        return _nvfp4_dequantize(qt)
    if qt.fmt.name == "fp8_e4m3":
        return _fp8_dequantize(qt)
    raise ValueError(f"unsupported format {qt.fmt.name}")


def fake_quant(x: torch.Tensor, fmt: str | QuantFormat = "mxfp4") -> torch.Tensor:
    """Quantize-dequantize and cast back to ``x.dtype`` (no STE here)."""
    qt = quantize(x, fmt)
    return dequantize(qt).to(x.dtype)


def fake_quant_fp8_amax(x: torch.Tensor, amax: torch.Tensor) -> torch.Tensor:
    """FP8 E4M3 fake-quant from a calibrated per-tensor ``amax``.

    Mirrors ModelOpt's ``tensor_quant_gpu_fp8.cu``: ``qscale = 448 / amax``,
    ``inv_scale = 1 / qscale`` in fp32, then ``fp8(x * qscale) * inv_scale``.
    Both divisions need float32 tensors on both sides; a Python float on the
    left selects an approximate CUDA divide and flips FP8 midpoints. Keeping
    ``amax`` (not ``amax / 448``) also avoids a 1-ulp round-trip loss that moved
    22/256 calibrated 27B modules.
    """
    amax = amax.to(device=x.device, dtype=torch.float32)
    safe_amax = torch.where(amax <= 2.0**-24, torch.ones_like(amax), amax)
    qscale = torch.full_like(safe_amax, E4M3_MAX) / safe_amax
    inv_scale = torch.full_like(qscale, 1.0) / qscale
    quantized = (x.float() * qscale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return (quantized.float() * inv_scale).to(x.dtype)


def fake_quant_fp8_static(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Legacy entry point where ``scale`` is ``amax / 448``.

    Prefer :func:`fake_quant_fp8_amax`: the scale round-trip can move amax by
    1 ulp (e.g. 0.248 -> 0.24800001).
    """
    return fake_quant_fp8_amax(x, scale.to(torch.float32) * 448.0)
