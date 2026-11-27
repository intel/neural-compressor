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
"""Fused single-pass Triton QDQ kernels for MXFP4, NVFP4, MXFP8, and FP8 E4M3.

Key properties:
1. **0-ULP bit-level parity** with ``functional.py`` (``"pytorch"`` backend) and
   NVIDIA ModelOpt CUDA kernels across random, boundary, midpoint, zero, and
   subnormal inputs by using IEEE-754 ``tl.div_rn``, NVIDIA ``libdevice``
   ``log2``/``ceil``/``exp2``, and exact register-predicated E2M1 round-to-even.
2. **Zero intermediate global memory buffers**: block-wise ``amax``, E8M0 / E4M3
   scale quantization, E2M1 / E4M3 element rounding, and rescaling all execute in
   registers within a single kernel pass (1 global read + 1 global write).
3. **64-bit pointer safety**: supports activations exceeding ``2**31`` elements
   (e.g., ``[B, 8192, 17408]`` on 27B+ models) without 32-bit index overflow.
"""

from __future__ import annotations

from typing import Optional

import torch

from .formats import QuantFormat

try:
    import triton
    import triton.language as tl

    try:
        from triton.language.extra import libdevice as _tl_libdevice
    except ImportError:  # pragma: no cover
        from triton.language.extra.cuda import libdevice as _tl_libdevice

    _TRITON_AVAILABLE = hasattr(tl, "float8e4nv")
except Exception:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _tl_libdevice = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False


_SUPPORTED_TRITON_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def is_triton_qdq_available() -> bool:
    """Return True when Triton, CUDA, and ``tl.float8e4nv`` are available."""
    return bool(_TRITON_AVAILABLE and torch.cuda.is_available())


if _TRITON_AVAILABLE:
    _INV_E2M1_MAX_F32 = tl.constexpr(1.0 / 6.0)
    _INV_E4M3_MAX_F32 = tl.constexpr(1.0 / 448.0)
    _INV_NVFP4_MAX_F32 = tl.constexpr(1.0 / 2688.0)

    @triton.jit
    def _fdiv_rn_noftz(a, b):
        return tl.inline_asm_elementwise(
            "div.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _fmul_rn_noftz(a, b):
        return tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _e8m0_scale_from_amax(amax, inv_max):
        """Compute exact E8M0 power-of-two scale (including subnormal 2^-127) without FTZ."""
        descale = _fmul_rn_noftz(amax, inv_max)
        # Lift subnormals by 2^24 before libdevice.log2 (which runs under FTZ in NVPTX).
        descale_up = _fmul_rn_noftz(descale, 16777216.0)
        e8m0 = _tl_libdevice.ceil(tl.maximum(_tl_libdevice.log2(descale_up) - 24.0, -127.0))
        e8m0 = tl.where(amax > 0.0, e8m0, -127.0)
        enc_i32 = tl.minimum(tl.maximum(e8m0 + 127.0, 0.0), 255.0).to(tl.int32)
        scale_bits = tl.where(enc_i32 == 0, 0x00400000, enc_i32 << 23)
        return scale_bits.to(tl.float32, bitcast=True)

    @triton.jit
    def _e2m1_round_even_f32(scaled):
        """Register-only IEEE round-half-to-even to E2M1 ``{0, 0.5, 1, 1.5, 2, 3, 4, 6}``."""
        absx = tl.abs(scaled)
        q_abs = tl.where(
            absx > 5.0,
            6.0,
            tl.where(
                absx >= 3.5,
                4.0,
                tl.where(
                    absx > 2.5,
                    3.0,
                    tl.where(
                        absx >= 1.75,
                        2.0,
                        tl.where(
                            absx > 1.25,
                            1.5,
                            tl.where(
                                absx >= 0.75,
                                1.0,
                                tl.where(absx > 0.25, 0.5, 0.0),
                            ),
                        ),
                    ),
                ),
            ),
        )
        return tl.where(scaled < 0.0, -q_abs, q_abs)

    @triton.jit
    def _mxfp4_qdq_kernel(
        x_ptr,
        out_ptr,
        n_blocks,
        BLOCKS_PER_PID: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        block_idx = pid * BLOCKS_PER_PID + tl.arange(0, BLOCKS_PER_PID)[:, None].to(tl.int64)
        elem_idx = tl.arange(0, 32)[None, :].to(tl.int64)
        mask = block_idx < n_blocks
        offs = block_idx * 32 + elem_idx

        x_raw = tl.load(x_ptr + offs, mask=mask, other=0.0)
        xf = x_raw.to(tl.float32)

        amax = tl.max(tl.abs(xf), axis=1, keep_dims=True)
        scale = _e8m0_scale_from_amax(amax, _INV_E2M1_MAX_F32)

        q = _e2m1_round_even_f32(_fdiv_rn_noftz(xf, scale))
        out = _fmul_rn_noftz(q, scale).to(x_raw.dtype)
        tl.store(out_ptr + offs, out, mask=mask)

    @triton.jit
    def _mxfp8_qdq_kernel(
        x_ptr,
        out_ptr,
        n_blocks,
        BLOCKS_PER_PID: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        block_idx = pid * BLOCKS_PER_PID + tl.arange(0, BLOCKS_PER_PID)[:, None].to(tl.int64)
        elem_idx = tl.arange(0, 32)[None, :].to(tl.int64)
        mask = block_idx < n_blocks
        offs = block_idx * 32 + elem_idx

        x_raw = tl.load(x_ptr + offs, mask=mask, other=0.0)
        xf = x_raw.to(tl.float32)

        amax = tl.max(tl.abs(xf), axis=1, keep_dims=True)
        scale = _e8m0_scale_from_amax(amax, _INV_E4M3_MAX_F32)

        scaled = tl.minimum(tl.maximum(_fdiv_rn_noftz(xf, scale), -448.0), 448.0)
        q_fp8 = scaled.to(tl.float8e4nv).to(tl.float32)
        out = _fmul_rn_noftz(q_fp8, scale).to(x_raw.dtype)
        tl.store(out_ptr + offs, out, mask=mask)

    @triton.jit
    def _nvfp4_qdq_kernel(
        x_ptr,
        mn_ptr,
        mx_ptr,
        out_ptr,
        n_blocks,
        BLOCKS_PER_PID: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        block_idx = pid * BLOCKS_PER_PID + tl.arange(0, BLOCKS_PER_PID)[:, None].to(tl.int64)
        elem_idx = tl.arange(0, 16)[None, :].to(tl.int64)
        mask = block_idx < n_blocks
        offs = block_idx * 16 + elem_idx

        mn = tl.load(mn_ptr).to(tl.float32)
        mx = tl.load(mx_ptr).to(tl.float32)
        global_amax = tl.maximum(tl.abs(mx), tl.abs(mn))
        raw_global = _fmul_rn_noftz(global_amax, _INV_NVFP4_MAX_F32)
        global_scale = tl.where(raw_global > 0.0, raw_global, 1.0)

        x_raw = tl.load(x_ptr + offs, mask=mask, other=0.0)
        xf = x_raw.to(tl.float32)

        amax = tl.max(tl.abs(xf), axis=1, keep_dims=True)
        denom = _fmul_rn_noftz(6.0, global_scale)
        block_scale = tl.minimum(tl.maximum(_fdiv_rn_noftz(amax, denom), 0.001953125), 448.0)
        block_scale = block_scale.to(tl.float8e4nv).to(tl.float32)
        eff_scale = _fmul_rn_noftz(block_scale, global_scale)

        q = _e2m1_round_even_f32(_fdiv_rn_noftz(xf, eff_scale))
        out = _fmul_rn_noftz(q, eff_scale).to(x_raw.dtype)
        tl.store(out_ptr + offs, out, mask=mask)

    @triton.jit
    def _fp8_static_qdq_kernel(
        x_ptr,
        amax_ptr,
        out_ptr,
        n_elements,
        BLOCK_ELEMS: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        offs = pid * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS).to(tl.int64)
        mask = offs < n_elements

        amax_f32 = tl.load(amax_ptr).to(tl.float32)
        safe_amax = tl.where(amax_f32 <= 5.960464477539063e-08, 1.0, amax_f32)  # 2**-24
        qscale = _fdiv_rn_noftz(448.0, safe_amax)
        inv_scale = _fdiv_rn_noftz(1.0, qscale)

        x_raw = tl.load(x_ptr + offs, mask=mask, other=0.0)
        xf = x_raw.to(tl.float32)
        scaled = tl.minimum(tl.maximum(_fmul_rn_noftz(xf, qscale), -448.0), 448.0)
        q_fp8 = scaled.to(tl.float8e4nv).to(tl.float32)
        out = _fmul_rn_noftz(q_fp8, inv_scale).to(x_raw.dtype)
        tl.store(out_ptr + offs, out, mask=mask)

    @triton.jit
    def _fp8_dynamic_qdq_kernel(
        x_ptr,
        mn_ptr,
        mx_ptr,
        out_ptr,
        n_elements,
        BLOCK_ELEMS: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        offs = pid * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS).to(tl.int64)
        mask = offs < n_elements

        mn = tl.load(mn_ptr).to(tl.float32)
        mx = tl.load(mx_ptr).to(tl.float32)
        amax = tl.maximum(tl.abs(mx), tl.abs(mn))
        scale = _fmul_rn_noftz(amax, _INV_E4M3_MAX_F32)
        safe_scale = tl.where(scale > 0.0, scale, 1.0)

        x_raw = tl.load(x_ptr + offs, mask=mask, other=0.0)
        xf = x_raw.to(tl.float32)
        scaled = tl.minimum(tl.maximum(_fdiv_rn_noftz(xf, safe_scale), -448.0), 448.0)
        q_fp8 = scaled.to(tl.float8e4nv).to(tl.float32)
        out = _fmul_rn_noftz(q_fp8, safe_scale).to(x_raw.dtype)
        tl.store(out_ptr + offs, out, mask=mask)


def triton_dynamic_qdq(
    x: torch.Tensor,
    fmt: QuantFormat,
    *,
    pytorch_fallback=None,
) -> torch.Tensor:
    """Execute fused single-pass Triton dynamic QDQ on CUDA tensors."""
    if fmt.name == "bf16":
        return x

    if (
        not _TRITON_AVAILABLE
        or x.device.type != "cuda"
        or x.dtype not in _SUPPORTED_TRITON_DTYPES
        or torch.compiler.is_compiling()
    ):
        if pytorch_fallback is not None:
            return pytorch_fallback(x, fmt)
        raise RuntimeError("Triton QDQ backend cannot execute on the given tensor/context.")

    if fmt.block_size is not None:
        if x.ndim < 1:
            raise ValueError(f"Block-scaled fake-quant expects at least 1-D input, got shape {tuple(x.shape)}")
        last_dim = x.shape[-1]
        if last_dim % fmt.block_size != 0:
            raise ValueError(f"Last dimension ({last_dim}) must be divisible by block_size ({fmt.block_size})")

    numel = x.numel()
    if numel == 0:
        return x.clone()

    x_contig = x if x.is_contiguous() else x.contiguous()
    out = torch.empty_like(x_contig)

    with torch.cuda.device(x_contig.device):
        if fmt.name == "mxfp4":
            n_blocks = numel // 32
            blocks_per_pid = 8
            grid = (triton.cdiv(n_blocks, blocks_per_pid),)
            _mxfp4_qdq_kernel[grid](
                x_contig,
                out,
                n_blocks,
                BLOCKS_PER_PID=blocks_per_pid,
                num_warps=4,
            )
        elif fmt.name == "mxfp8":
            n_blocks = numel // 32
            blocks_per_pid = 8
            grid = (triton.cdiv(n_blocks, blocks_per_pid),)
            _mxfp8_qdq_kernel[grid](
                x_contig,
                out,
                n_blocks,
                BLOCKS_PER_PID=blocks_per_pid,
                num_warps=4,
            )
        elif fmt.name == "nvfp4":
            mn, mx = torch.aminmax(x_contig)
            n_blocks = numel // 16
            blocks_per_pid = 16
            grid = (triton.cdiv(n_blocks, blocks_per_pid),)
            _nvfp4_qdq_kernel[grid](
                x_contig,
                mn,
                mx,
                out,
                n_blocks,
                BLOCKS_PER_PID=blocks_per_pid,
                num_warps=4,
            )
        elif fmt.name == "fp8_e4m3":
            mn, mx = torch.aminmax(x_contig)
            block_elems = 1024
            grid = (triton.cdiv(numel, block_elems),)
            _fp8_dynamic_qdq_kernel[grid](
                x_contig,
                mn,
                mx,
                out,
                numel,
                BLOCK_ELEMS=block_elems,
                num_warps=4,
            )
        else:
            raise ValueError(f"Unsupported dynamic format {fmt.name!r}")

    return out


def triton_fp8_static_amax_qdq(
    x: torch.Tensor,
    amax: torch.Tensor,
    *,
    pytorch_fallback=None,
) -> torch.Tensor:
    """Execute fused single-pass Triton static FP8 E4M3 QDQ matching ModelOpt CUDA."""
    if (
        not _TRITON_AVAILABLE
        or x.device.type != "cuda"
        or x.dtype not in _SUPPORTED_TRITON_DTYPES
        or torch.compiler.is_compiling()
    ):
        if pytorch_fallback is not None:
            return pytorch_fallback(x, amax)
        raise RuntimeError("Triton QDQ backend cannot execute on the given tensor/context.")

    numel = x.numel()
    if numel == 0:
        return x.clone()

    x_contig = x if x.is_contiguous() else x.contiguous()
    amax_dev: Optional[torch.Tensor] = amax if amax.device == x_contig.device else amax.to(device=x_contig.device)
    assert amax_dev is not None
    if not amax_dev.is_contiguous():
        amax_dev = amax_dev.contiguous()
    out = torch.empty_like(x_contig)

    block_elems = 1024
    grid = (triton.cdiv(numel, block_elems),)
    with torch.cuda.device(x_contig.device):
        _fp8_static_qdq_kernel[grid](
            x_contig,
            amax_dev,
            out,
            numel,
            BLOCK_ELEMS=block_elems,
            num_warps=4,
        )
    return out


__all__ = [
    "is_triton_qdq_available",
    "triton_dynamic_qdq",
    "triton_fp8_static_amax_qdq",
]
