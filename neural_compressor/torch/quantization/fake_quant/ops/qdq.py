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
"""Hot-path Quantize-Dequantize (QDQ) kernels, STE autograd wrapper, and backend registry.

Design invariants for training hot paths:
1. Zero bit-packing / unpacking or ``repeat_interleave`` expansion during QDQ.
2. Per-device lookup-table caching (zero CPU-to-device H2D transfers per forward).
3. Zero host-device synchronization (no ``float(tensor)`` / ``.item()`` branches),
   enabling ``torch.compile(..., fullgraph=True)`` with zero graph breaks.
4. Zero saved tensors in :class:`StraightThroughFakeQuant` backward.
5. Exact bit-level numerical parity with ModelOpt's CUDA MXFP4, NVFP4, and FP8 E4M3 kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import torch

from ..formats import (
    E2M1_BOUNDS,
    E2M1_MAX,
    E2M1_VALUES,
    E4M3_MAX,
    QuantFormat,
    get_format,
)

QuantBackendName = Literal["auto", "pytorch", "triton"]

# Bound peak FP32/int64 scratch memory to ~160 MiB even on [B, 8192, 17408] activations.
_MAX_CHUNK_ELEMENTS = 8_388_608

_DEVICE_LUT_CACHE: dict[tuple[str, Optional[int]], tuple[torch.Tensor, torch.Tensor]] = {}


def _get_e2m1_luts(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cached ``(bounds_f32, values_f32)`` on ``device`` without H2D copies."""
    key = (device.type, device.index)
    if not torch.compiler.is_compiling():
        cached = _DEVICE_LUT_CACHE.get(key)
        if cached is not None:
            return cached
    bounds = torch.tensor(E2M1_BOUNDS, dtype=torch.float32, device=device)
    values = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=device)
    if not torch.compiler.is_compiling():
        _DEVICE_LUT_CACHE[key] = (bounds, values)
    return bounds, values


def _check_block_dim(x: torch.Tensor, block_size: int) -> None:
    if x.ndim < 1:
        raise ValueError(f"Block-scaled fake-quant expects at least 1-D input, got shape {tuple(x.shape)}")
    last_dim = x.shape[-1]
    if last_dim % block_size != 0:
        raise ValueError(f"Last dimension ({last_dim}) must be divisible by block_size ({block_size})")


def _e2m1_qdq_scaled(scaled: torch.Tensor) -> torch.Tensor:
    """Round ``scaled`` to E2M1 with IEEE round-half-to-even at odd midpoints ``[0.75, 1.75, 3.5]``."""
    bounds, values = _get_e2m1_luts(scaled.device)
    absx = scaled.abs()
    order = torch.searchsorted(bounds, absx)
    # 0.75, 1.75, and 3.5 are exact dyadic floats in IEEE-754 FP32.
    tie = ((absx == 0.75) | (absx == 1.75) | (absx == 3.5)).to(order.dtype)
    q_abs = values[order + tie]
    return torch.where(scaled < 0, -q_abs, q_abs)


def _mxfp4_qdq_block(blocks_f32: torch.Tensor) -> torch.Tensor:
    amax = blocks_f32.abs().amax(dim=-1, keepdim=True)
    descale = amax / E2M1_MAX
    e8m0 = torch.ceil(torch.maximum(torch.log2(descale), torch.full_like(descale, -127.0)))
    e8m0 = torch.where(amax > 0, e8m0, torch.full_like(e8m0, -127.0))
    # Clamp encoded E8M0 exponent to uint8 [0, 255] -> unbiased [-127, 128].
    exp_i32 = ((e8m0 + 127.0).clamp(0.0, 255.0) - 127.0).to(torch.int32)
    scale = torch.ldexp(torch.ones_like(e8m0), exp_i32)
    q = _e2m1_qdq_scaled(blocks_f32 / scale)
    return q * scale


def _mxfp8_qdq_block(blocks_f32: torch.Tensor) -> torch.Tensor:
    amax = blocks_f32.abs().amax(dim=-1, keepdim=True)
    descale = amax / E4M3_MAX
    e8m0 = torch.ceil(torch.maximum(torch.log2(descale), torch.full_like(descale, -127.0)))
    e8m0 = torch.where(amax > 0, e8m0, torch.full_like(e8m0, -127.0))
    exp_i32 = ((e8m0 + 127.0).clamp(0.0, 255.0) - 127.0).to(torch.int32)
    scale = torch.ldexp(torch.ones_like(e8m0), exp_i32)
    scaled = (blocks_f32 / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float()
    return scaled * scale


def _nvfp4_qdq_block(blocks_f32: torch.Tensor, global_scale: torch.Tensor) -> torch.Tensor:
    amax = blocks_f32.abs().amax(dim=-1, keepdim=True)
    block_scale = (amax / (E2M1_MAX * global_scale)).clamp(2.0**-9, E4M3_MAX)
    block_scale = block_scale.to(torch.float8_e4m3fn).float()
    eff_scale = block_scale * global_scale
    q = _e2m1_qdq_scaled(blocks_f32 / eff_scale)
    return q * eff_scale


def _pytorch_dynamic_qdq(x: torch.Tensor, fmt: QuantFormat) -> torch.Tensor:
    """Pure-PyTorch vectorized QDQ without nibble packing or host syncs."""
    if fmt.name == "bf16":
        return x
    if fmt.name == "fp8_e4m3":
        xf = x.float()
        amax = xf.abs().amax()
        scale = amax / E4M3_MAX
        safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        q = (xf / safe_scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        return (q.float() * safe_scale).to(x.dtype)

    block_size = fmt.block_size
    if block_size is None:
        raise ValueError(f"Unsupported dynamic format {fmt.name!r}")
    _check_block_dim(x, block_size)

    numel = x.numel()
    if numel == 0:
        return x.clone()

    global_scale: Optional[torch.Tensor] = None
    if fmt.name == "nvfp4":
        global_amax = x.abs().amax().float()
        raw_global = global_amax / (E2M1_MAX * E4M3_MAX)
        global_scale = torch.where(raw_global > 0, raw_global, torch.ones_like(raw_global))

    if torch.compiler.is_compiling() or numel <= _MAX_CHUNK_ELEMENTS:
        blocks = x.float().reshape(-1, block_size)
        if fmt.name == "mxfp4":
            out_blocks = _mxfp4_qdq_block(blocks)
        elif fmt.name == "nvfp4":
            assert global_scale is not None
            out_blocks = _nvfp4_qdq_block(blocks, global_scale)
        elif fmt.name == "mxfp8":
            out_blocks = _mxfp8_qdq_block(blocks)
        else:
            raise ValueError(f"Unsupported block format {fmt.name!r}")
        return out_blocks.reshape(x.shape).to(x.dtype)

    flat_in = x.reshape(-1, block_size)
    flat_out = torch.empty_like(flat_in)
    blocks_per_chunk = max(_MAX_CHUNK_ELEMENTS // block_size, 1)
    total_blocks = flat_in.shape[0]
    for start in range(0, total_blocks, blocks_per_chunk):
        end = min(start + blocks_per_chunk, total_blocks)
        chunk_f32 = flat_in[start:end].float()
        if fmt.name == "mxfp4":
            q_chunk = _mxfp4_qdq_block(chunk_f32)
        elif fmt.name == "nvfp4":
            assert global_scale is not None
            q_chunk = _nvfp4_qdq_block(chunk_f32, global_scale)
        elif fmt.name == "mxfp8":
            q_chunk = _mxfp8_qdq_block(chunk_f32)
        else:
            raise ValueError(f"Unsupported block format {fmt.name!r}")
        flat_out[start:end] = q_chunk.to(x.dtype)
    return flat_out.reshape(x.shape)


def _pytorch_fp8_static_amax_qdq(x: torch.Tensor, amax: torch.Tensor) -> torch.Tensor:
    """Static per-tensor FP8 E4M3 QDQ matching ModelOpt's ``tensor_quant_gpu_fp8.cu``.

    Both divisions use FP32 tensors on both operands so CUDA selects IEEE-compliant
    ``div.rn.f32`` rather than approximate scalar-reciprocal instructions.
    """
    amax_f32 = amax.to(device=x.device, dtype=torch.float32)
    safe_amax = torch.where(amax_f32 <= 2.0**-24, torch.ones_like(amax_f32), amax_f32)
    qscale = torch.full_like(safe_amax, E4M3_MAX) / safe_amax
    inv_scale = torch.full_like(qscale, 1.0) / qscale
    quantized = (x.float() * qscale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return (quantized.float() * inv_scale).to(x.dtype)


# ---------------------------------------------------------------------------
# Pluggable QDQ Backend Registry
# ---------------------------------------------------------------------------


@dataclass
class _BackendEntry:
    name: str
    dynamic_fn: Callable[[torch.Tensor, QuantFormat], torch.Tensor]
    fp8_static_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    is_available_fn: Callable[[], bool]
    priority: int


_BACKENDS: dict[str, _BackendEntry] = {}
_DEFAULT_BACKEND: QuantBackendName = "auto"


def register_quant_backend(
    name: str,
    *,
    dynamic_fn: Callable[[torch.Tensor, QuantFormat], torch.Tensor],
    fp8_static_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    is_available_fn: Callable[[], bool] = lambda: True,
    priority: int = 0,
) -> None:
    """Register a QDQ kernel backend (e.g. ``"pytorch"`` or ``"triton"``)."""
    _BACKENDS[name] = _BackendEntry(
        name=name,
        dynamic_fn=dynamic_fn,
        fp8_static_fn=fp8_static_fn,
        is_available_fn=is_available_fn,
        priority=priority,
    )


def set_quant_backend(backend: QuantBackendName) -> None:
    """Set the process-wide default QDQ backend (``"auto"``, ``"pytorch"``, or ``"triton"``)."""
    global _DEFAULT_BACKEND
    if backend != "auto" and backend not in _BACKENDS:
        raise ValueError(f"Unknown quant backend {backend!r}; registered: {sorted(_BACKENDS)}")
    _DEFAULT_BACKEND = backend


def get_quant_backend() -> str:
    """Return the currently configured default QDQ backend name."""
    return _DEFAULT_BACKEND


def resolve_quant_backend(
    x_or_device: torch.Tensor | torch.device,
    backend: QuantBackendName | None = "auto",
) -> str:
    """Return the concrete QDQ backend name (``"pytorch"`` or ``"triton"``) selected for ``x_or_device``."""
    dev = x_or_device.device if isinstance(x_or_device, torch.Tensor) else x_or_device
    return _resolve_backend(backend or "auto", dev).name


def _resolve_backend(backend: QuantBackendName, device: torch.device) -> _BackendEntry:
    chosen = _DEFAULT_BACKEND if backend == "auto" else backend
    if chosen == "auto":
        # Prefer highest-priority available backend; non-CUDA devices always use pytorch unless supported.
        for entry in sorted(_BACKENDS.values(), key=lambda item: item.priority, reverse=True):
            if entry.name == "triton" and device.type != "cuda":
                continue
            if entry.is_available_fn():
                return entry
        return _BACKENDS["pytorch"]
    entry = _BACKENDS.get(chosen)
    if entry is None:
        raise ValueError(f"Unknown quant backend {chosen!r}; registered: {sorted(_BACKENDS)}")
    if not entry.is_available_fn():
        raise RuntimeError(f"Quant backend {chosen!r} is not available in the current environment")
    return entry


from .triton_kernels import (
    is_triton_qdq_available,
    triton_dynamic_qdq,
    triton_fp8_static_amax_qdq,
)

register_quant_backend(
    "pytorch",
    dynamic_fn=_pytorch_dynamic_qdq,
    fp8_static_fn=_pytorch_fp8_static_amax_qdq,
    is_available_fn=lambda: True,
    priority=10,
)

register_quant_backend(
    "triton",
    dynamic_fn=lambda x, fmt: triton_dynamic_qdq(x, fmt, pytorch_fallback=_pytorch_dynamic_qdq),
    fp8_static_fn=lambda x, amax: triton_fp8_static_amax_qdq(
        x, amax, pytorch_fallback=_pytorch_fp8_static_amax_qdq
    ),
    is_available_fn=is_triton_qdq_available,
    priority=20,
)


# ---------------------------------------------------------------------------
# Public Functional & STE Autograd Entry Points
# ---------------------------------------------------------------------------


def fake_quant_dynamic(
    x: torch.Tensor,
    fmt: str | QuantFormat = "mxfp4",
    *,
    backend: QuantBackendName = "auto",
) -> torch.Tensor:
    """Quantize-dequantize ``x`` dynamically without autograd tracking."""
    spec = get_format(fmt)
    if spec.name == "bf16":
        return x
    entry = _resolve_backend(backend, x.device)
    with torch.no_grad():
        return entry.dynamic_fn(x, spec)


def fake_quant_fp8_static_amax(
    x: torch.Tensor,
    amax: torch.Tensor,
    *,
    backend: QuantBackendName = "auto",
) -> torch.Tensor:
    """Quantize-dequantize ``x`` to FP8 E4M3 using calibrated per-tensor ``amax``."""
    entry = _resolve_backend(backend, x.device)
    with torch.no_grad():
        return entry.fp8_static_fn(x, amax)


class StraightThroughFakeQuant(torch.autograd.Function):
    """Zero-saved-tensor Straight-Through Estimator (STE) for fake quantization."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        fmt: QuantFormat,
        amax: Optional[torch.Tensor],
        dynamic: bool,
        backend: str,
    ) -> torch.Tensor:
        entry = _resolve_backend(backend, x.device)  # type: ignore[arg-type]
        if dynamic:
            return entry.dynamic_fn(x, fmt)
        if amax is None:
            raise RuntimeError(f"Static quantizer for {fmt.name} requires a calibrated amax tensor")
        return entry.fp8_static_fn(x, amax)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None, None, None, None


def ste_fake_quant(
    x: torch.Tensor,
    fmt: str | QuantFormat,
    amax: Optional[torch.Tensor] = None,
    *,
    dynamic: bool = True,
    backend: QuantBackendName = "auto",
) -> torch.Tensor:
    """Apply fake quantization to ``x`` with an identity STE backward."""
    spec = get_format(fmt)
    if spec.name == "bf16":
        return x
    return StraightThroughFakeQuant.apply(x, spec, amax, dynamic, backend)


__all__ = [
    "QuantBackendName",
    "StraightThroughFakeQuant",
    "fake_quant_dynamic",
    "fake_quant_fp8_static_amax",
    "get_quant_backend",
    "register_quant_backend",
    "set_quant_backend",
    "ste_fake_quant",
]
