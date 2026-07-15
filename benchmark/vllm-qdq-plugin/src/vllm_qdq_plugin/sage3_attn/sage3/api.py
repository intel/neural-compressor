"""Entry points for the sage3 standalone attention implementation.

Provides both backward-compatible API (sageattn3_torch_triton_standalone)
and the new config-based API (sageattn3_standalone).
"""

import math
import os
import warnings
from typing import Optional, Union

import torch
import torch.nn.functional as F
import triton.language as tl

from .attention_kernel import launch_attention
from .quant_config import ATTENTION_CONFIGS, AttentionConfig
from .quantize import quantize_qk, quantize_v
from .transforms import TransformContext, qk_smoothing

# ── Constants ──

SAGE3_TILE_SIZE = 128  # Fixed tile size — P-quant kernels require 128

ACC_DTYPE_MAP = {
    "fp32": (tl.float32, tl.float32, tl.float32),
    "fp16_pv_only": (tl.float32, tl.float16, tl.float32),
    "bf16_pv_only": (tl.float32, tl.bfloat16, tl.float32),
    "fp16_qk_only": (tl.float16, tl.float32, tl.float32),
    "bf16_qk_only": (tl.bfloat16, tl.float32, tl.float32),
    "fp16_both_dot": (tl.float16, tl.float16, tl.float32),
    "bf16_both_dot": (tl.bfloat16, tl.bfloat16, tl.float32),
    "fp16_full": (tl.float16, tl.float16, tl.float16),
    "bf16_full": (tl.bfloat16, tl.bfloat16, tl.bfloat16),
}


# ── Lazy environment variable helpers ──


def _get_env_bool(name: str, default: str = "0") -> bool:
    """Read a boolean env var at call time (not import time)."""
    return os.getenv(name, default).lower() in ("1", "true")


def _get_env_str(name: str, default: str) -> str:
    """Read a string env var at call time (not import time)."""
    return os.getenv(name, default).lower()


def debug_print(*args, **kwargs):
    """Debug print that respects SAGE3_DEBUG setting."""
    if _get_env_bool("SAGE3_DEBUG"):
        print("[SAGE3]", *args, **kwargs)


import functools


@functools.lru_cache(maxsize=None)
def print_once(*args, **kwargs):
    """Print a message only the first time it's encountered."""
    print(*args, **kwargs)


def _validate_runtime_config(
    config: AttentionConfig,
    d: int,
    tile_size_q: int,
    tile_size_k: int,
    acc_dtype: str,
) -> None:
    """Reject known-unrunnable runtime combinations with an actionable error."""
    if config.name == "nvfp4" and d == 128 and tile_size_q == 128 and tile_size_k == 128 and acc_dtype == "fp32":
        raise RuntimeError(
            "sage3_refactored NVFP4 with head_dim=128 and 128x128 tiles exceeds "
            "the Triton shared-memory limit in the default fp32 accumulator mode. "
            "Rerun with acc_dtype='bf16_both_dot' "
            "(for example: --attention_type sage3_refactored --split_mask "
            "--acc_dtype bf16_both_dot)."
        )


# ============================================================================
# Core implementation (shared by both APIs)
# ============================================================================


def _run_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    config: AttentionConfig,
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    per_block_mean: bool = True,
    tile_size_q: int = 128,
    tile_size_k: int = 128,
    return_lse: bool = False,
    debug: bool = False,
    acc_dtype: str = "fp32",
) -> torch.Tensor:
    """Core attention implementation.

    1. Run pre_transforms pipeline
    2. Quantize Q/K and V
    3. Launch kernel with P_QUANT_FN
    4. Apply post-transform corrections
    """
    B, H, N, D = q.shape
    original_seq_len = N

    # ── Hardware MXFP8 early dispatch (uses tl.dot_scaled, no host-side quant) ──
    if config.name == "mxfp8_hw":
        from .mxfp8_hw_kernel import mxfp8_flash_attention, quantize_to_mxfp8

        print_once(f"[SAGE3 TRACE] mxfp8_hw kernel dispatched: Q={q.shape}, K={k.shape}, V={v.shape}, D={D}")

        # Apply QK smoothing for delta_s correction (reduces quant error from outliers)
        disable_per_block_mean = _get_env_bool("SAGE3_DISABLE_PER_BLOCK_MEAN")
        delta_s = None
        if per_block_mean and not disable_per_block_mean:
            ctx = TransformContext()
            q, k, v, ctx = qk_smoothing(q, k, v, ctx)
            delta_s = ctx.delta_s  # [B, H, num_groups, N] float32

        # Pad sequence to BLOCK_M=128
        if N % 128 != 0:
            pad_len = 128 - (N % 128)
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
            B, H, N, D = q.shape
            # Pad delta_s along the N dimension to match
            if delta_s is not None:
                delta_s = F.pad(delta_s, (0, pad_len))

        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(D)

        # Quantize Q, K to MXFP8 (groups along HEAD_DIM)
        q_fp8, q_scale = quantize_to_mxfp8(q)
        k_fp8, k_scale = quantize_to_mxfp8(k)
        # V needs grouping along seq dim: permute to [B,H,D,N], quantize, permute back
        v_t = v.permute(0, 1, 3, 2).contiguous()  # [B,H,D,N]
        v_fp8_t, v_scale = quantize_to_mxfp8(v_t)  # v_scale: [B,H,D,N//32]
        v_fp8 = v_fp8_t.permute(0, 1, 3, 2).contiguous()  # [B,H,N,D]

        output = mxfp8_flash_attention(
            q_fp8,
            k_fp8,
            v_fp8,
            q_scale,
            k_scale,
            v_scale,
            causal=is_causal,
            sm_scale=sm_scale,
            delta_s=delta_s,
        )

        if output.size(2) != original_seq_len:
            output = output[:, :, :original_seq_len, :].contiguous()
        return output.to(q.dtype)

    # ── Hardware MXFP4 early dispatch (uses tl.dot_scaled e2m1) ──
    if config.name == "mxfp4_hw":
        from .mxfp4_hw_kernel import mxfp4_flash_attention, quantize_to_mxfp4

        print_once(f"[SAGE3 TRACE] mxfp4_hw kernel dispatched: Q={q.shape}, K={k.shape}, V={v.shape}, D={D}")

        # Apply QK smoothing for delta_s correction (reduces FP4 quant error ~5×)
        disable_per_block_mean = _get_env_bool("SAGE3_DISABLE_PER_BLOCK_MEAN")
        delta_s = None
        if per_block_mean and not disable_per_block_mean:
            ctx = TransformContext()
            q, k, v, ctx = qk_smoothing(q, k, v, ctx)
            delta_s = ctx.delta_s  # [B, H, num_groups, N] float32

        # Pad sequence to BLOCK_M=128
        if N % 128 != 0:
            pad_len = 128 - (N % 128)
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
            B, H, N, D = q.shape
            # Pad delta_s along the N dimension to match
            if delta_s is not None:
                delta_s = F.pad(delta_s, (0, pad_len))

        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(D)

        # Quantize Q, K to MXFP4 (packed along HEAD_DIM) -> [B,H,N,D//2] uint8
        q_packed, q_scale = quantize_to_mxfp4(q)
        k_packed, k_scale = quantize_to_mxfp4(k)
        # V needs packing along seq dim: permute to [B,H,D,N], quantize, get [B,H,D,N//2]
        v_t = v.permute(0, 1, 3, 2).contiguous()  # [B,H,D,N]
        v_flat = v_t.reshape(-1, N)
        v_packed_flat, v_scale_flat = quantize_to_mxfp4(v_flat)
        v_packed = v_packed_flat.reshape(B, H, D, N // 2)  # [B,H,D,N//2]
        v_scale = v_scale_flat.reshape(B, H, D, N // 32)  # [B,H,D,N//32]

        output = mxfp4_flash_attention(
            q_packed,
            k_packed,
            v_packed,
            q_scale,
            k_scale,
            v_scale,
            causal=is_causal,
            sm_scale=sm_scale,
            delta_s=delta_s,
        )

        if output.size(2) != original_seq_len:
            output = output[:, :, :original_seq_len, :].contiguous()
        return output.to(q.dtype)

    # ── Mixed MXFP8 QK + MXFP4 PV early dispatch ──
    if config.name == "mixed_mxfp8qk_mxfp4pv_hw":
        from .mixed_mxfp8qk_mxfp4pv_hw_kernel import (
            mixed_mxfp8qk_mxfp4pv_flash_attention,
            quantize_to_mxfp4,
            quantize_to_mxfp8,
        )

        print_once(
            f"[SAGE3 TRACE] mixed_mxfp8qk_mxfp4pv_hw kernel dispatched: Q={q.shape}, K={k.shape}, V={v.shape}, D={D}"
        )

        # Pad sequence to BLOCK_M=128
        if N % 128 != 0:
            pad_len = 128 - (N % 128)
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
            B, H, N, D = q.shape

        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(D)

        # Quantize Q, K to MXFP8 (groups along HEAD_DIM)
        q_fp8, q_scale = quantize_to_mxfp8(q)
        k_fp8, k_scale = quantize_to_mxfp8(k)

        # Quantize V to MXFP4 (along sequence dim N, col-major)
        v_t = v.permute(0, 1, 3, 2).contiguous()  # [B,H,D,N]
        v_flat = v_t.reshape(-1, N)
        v_packed_flat, v_scale_flat = quantize_to_mxfp4(v_flat)
        v_packed = v_packed_flat.reshape(B, H, D, N // 2)  # [B,H,D,N//2]
        v_scale = v_scale_flat.reshape(B, H, D, N // 32)  # [B,H,D,N//32]

        output = mixed_mxfp8qk_mxfp4pv_flash_attention(
            q_fp8,
            k_fp8,
            v_packed,
            q_scale,
            k_scale,
            v_scale,
            causal=is_causal,
            sm_scale=sm_scale,
        )

        if output.size(2) != original_seq_len:
            output = output[:, :, :original_seq_len, :].contiguous()
        return output.to(q.dtype)

    # ── Mixed MXFP4 QK + MXFP8 PV early dispatch ──
    if config.name == "mixed_mxfp4qk_mxfp8pv_hw":
        from .mixed_mxfp4qk_mxfp8pv_hw_kernel import (
            mixed_mxfp4qk_mxfp8pv_flash_attention,
            quantize_to_mxfp4,
            quantize_to_mxfp8,
        )

        print_once(
            f"[SAGE3 TRACE] mixed_mxfp4qk_mxfp8pv_hw kernel dispatched: Q={q.shape}, K={k.shape}, V={v.shape}, D={D}"
        )

        # Pad sequence to BLOCK_M=128
        if N % 128 != 0:
            pad_len = 128 - (N % 128)
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
            B, H, N, D = q.shape

        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(D)

        # Quantize Q, K to MXFP4 (packed along HEAD_DIM) -> [B,H,N,D//2] uint8
        q_packed, q_scale = quantize_to_mxfp4(q)
        k_packed, k_scale = quantize_to_mxfp4(k)

        # Quantize V to MXFP8 (along sequence dim N)
        # V_scale shape: [B, H, D, N//32] — groups of 32 along sequence
        v_t = v.permute(0, 1, 3, 2).contiguous()  # [B,H,D,N]
        v_flat = v_t.reshape(-1, N)
        v_fp8_flat, v_scale_flat = quantize_to_mxfp8(v_flat)
        v_fp8_transposed = v_fp8_flat.reshape(B, H, D, N)  # [B,H,D,N]
        v_scale = v_scale_flat.reshape(B, H, D, N // 32)  # [B,H,D,N//32]
        # V data for kernel: [B, H, N, D] (standard layout)
        v_fp8 = v_fp8_transposed.permute(0, 1, 3, 2).contiguous()  # [B,H,N,D]

        output = mixed_mxfp4qk_mxfp8pv_flash_attention(
            q_packed,
            k_packed,
            v_fp8,
            q_scale,
            k_scale,
            v_scale,
            causal=is_causal,
            sm_scale=sm_scale,
        )

        if output.size(2) != original_seq_len:
            output = output[:, :, :original_seq_len, :].contiguous()
        return output.to(q.dtype)

    # Pad sequence to a multiple of tile_size so that every Triton tile is full.
    # Without this, the last partial tile computes out-of-bounds pointers (even
    # for masked lanes), which causes cudaErrorIllegalAddress on Blackwell (SM 10.0)
    # and is undefined behaviour on other architectures.
    tile_size = max(tile_size_q, tile_size_k)
    if N % tile_size != 0:
        pad_len = tile_size - (N % tile_size)
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        B, H, N, D = q.shape  # update N to padded length

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    if acc_dtype not in ACC_DTYPE_MAP:
        raise ValueError(f"Unknown acc_dtype '{acc_dtype}'. " f"Available: {list(ACC_DTYPE_MAP.keys())}")

    _validate_runtime_config(config, D, tile_size_q, tile_size_k, acc_dtype)

    qk_dot_dtype, pv_dot_dtype, softmax_dtype = ACC_DTYPE_MAP[acc_dtype]

    if debug:
        debug_print(f"Input shapes - Q: {q.shape}, K: {k.shape}, V: {v.shape}")
        debug_print(f"Config: {config.name}, tile Q={tile_size_q}, K={tile_size_k}")
        debug_print(f"Accumulator dtype mode: {acc_dtype}")

    # Step 1: Pre-transforms pipeline
    ctx = TransformContext()

    disable_per_block_mean = _get_env_bool("SAGE3_DISABLE_PER_BLOCK_MEAN")

    for transform in config.pre_transforms:
        # per_block_mean=False only skips QK smoothing, not other transforms
        # (e.g., Hadamard rotation should still run)
        if (not per_block_mean or disable_per_block_mean) and transform is qk_smoothing:
            if debug:
                debug_print("Skipping QK smoothing (per_block_mean=False)")
            continue
        q, k, v, ctx = transform(q, k, v, ctx)
        if debug:
            debug_print(f"Applied transform, delta_s: {ctx.delta_s is not None}")

    print_once(f"config: {config}")

    # Step 2: Quantize Q/K and V
    q_quant, _ = quantize_qk(q, config.qk_quant)
    k_quant, _ = quantize_qk(k, config.qk_quant)
    v_quant, _ = quantize_v(v, config.pv_quant)

    # Step 3: Launch attention kernel
    if debug:
        debug_print(f"Launching kernel (p_quant_fn={config.p_quant_fn})")
    num_warps = int(os.environ.get("num_warps", 8))
    output = launch_attention(
        q_quant,
        k_quant,
        v_quant,
        delta_s=ctx.delta_s,
        sm_scale=sm_scale,
        is_causal=is_causal,
        p_quant_fn=config.p_quant_fn,
        tile_size_q=tile_size_q,
        tile_size_k=tile_size_k,
        qk_dot_dtype=qk_dot_dtype,
        pv_dot_dtype=pv_dot_dtype,
        softmax_dtype=softmax_dtype,
        num_warps=num_warps,
    )

    # Step 4: Post-transform corrections
    # V-smoothing correction: output += v_mean (broadcast)
    if ctx.v_mean is not None:
        # The kernel computed attention over (v - v_mean), so we need to add
        # v_mean back: output = sum(softmax * (v - v_mean)) + v_mean
        output = output + ctx.v_mean

    if debug:
        debug_print(f"Output shape: {output.shape}")
        debug_print(f"Output range: [{output.min().item():.6f}, {output.max().item():.6f}]")

    # Step 5: Trim back to original sequence length if needed
    if output.size(2) != original_seq_len:
        output = output[:, :, :original_seq_len, :].contiguous()

    # Ensure output dtype matches input dtype
    if output.dtype != q.dtype:
        output = output.to(q.dtype)

    if return_lse:
        raise NotImplementedError("return_lse is not yet supported by the sage3 composable implementation")
    return output


# ============================================================================
# Backward-compatible entry point
# ============================================================================


@torch.inference_mode()
def sageattn3_torch_triton_standalone(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    per_block_mean: bool = True,
    tile_size_q: int = 128,
    tile_size_k: int = 128,
    return_lse: bool = False,
    debug: bool = False,
    quant_format: str = "mxfp4",
    acc_dtype: str = "fp32",
):
    """SageAttention3 Triton implementation — backward-compatible API.

    Same signature as the original monolith's sageattn3_torch_triton_standalone.

    Args:
        q, k, v: Input tensors [B, H, N, D]
        tensor_layout: Must be "HND"
        is_causal: Apply causal masking
        sm_scale: Softmax scale (default: 1/sqrt(D))
        per_block_mean: Apply QK smoothing
        tile_size_q: Query tile size
        tile_size_k: Key/Value tile size
        return_lse: Return log-sum-exp (not implemented)
        debug: Enable debug logging
        quant_format: Format name ('nvfp4', 'mxfp4', 'mxfp4_s1', 'mxfp8_s1')
        acc_dtype: Accumulator mode name
    """
    if tensor_layout != "HND":
        raise ValueError("Only HND tensor layout is supported")

    if quant_format not in ATTENTION_CONFIGS:
        raise ValueError(f"Unknown quant_format '{quant_format}'. " f"Available: {list(ATTENTION_CONFIGS.keys())}")

    config = ATTENTION_CONFIGS[quant_format]
    return _run_attention(
        q,
        k,
        v,
        config,
        is_causal=is_causal,
        sm_scale=sm_scale,
        per_block_mean=per_block_mean,
        tile_size_q=tile_size_q,
        tile_size_k=tile_size_k,
        return_lse=return_lse,
        debug=debug,
        acc_dtype=acc_dtype,
    )


# ============================================================================
# New config-based entry point
# ============================================================================


@torch.inference_mode()
def sageattn3_standalone(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    config: Union[str, AttentionConfig] = "mxfp4",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    per_block_mean: bool = True,
    tile_size_q: int = 128,
    tile_size_k: int = 128,
    return_lse: bool = False,
    debug: bool = False,
    acc_dtype: str = "fp32",
):
    """SageAttention3 Triton implementation — new config-based API.

    Accepts either a config name string or an AttentionConfig object.
    """
    if isinstance(config, str):
        if config not in ATTENTION_CONFIGS:
            raise ValueError(f"Unknown config '{config}'. " f"Available: {list(ATTENTION_CONFIGS.keys())}")
        config = ATTENTION_CONFIGS[config]

    return _run_attention(
        q,
        k,
        v,
        config,
        is_causal=is_causal,
        sm_scale=sm_scale,
        per_block_mean=per_block_mean,
        tile_size_q=tile_size_q,
        tile_size_k=tile_size_k,
        return_lse=return_lse,
        debug=debug,
        acc_dtype=acc_dtype,
    )


# ============================================================================
# SDPA-Compatible Wrapper
# ============================================================================


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    **kwargs,
) -> torch.Tensor:
    """SDPA-compatible wrapper for SageAttention3 Triton implementation.

    Limitations vs torch.nn.functional.scaled_dot_product_attention:
    - Q, K, V must have identical shapes [B, H, N, D] (no cross-attention)
    - dropout_p is not supported (silently ignored with warning)
    - attn_mask is not supported (silently ignored with warning)
    - KV caching (different N for K/V) is not supported
    """
    if not query.is_cuda:
        raise RuntimeError("SageAttention3 requires CUDA tensors")

    if dropout_p > 0.0:
        warnings.warn(
            f"SageAttention3 doesn't support dropout_p={dropout_p}, ignoring",
            UserWarning,
            stacklevel=2,
        )

    if attn_mask is not None:
        warnings.warn(
            "SageAttention3 doesn't support arbitrary attention masks, ignoring",
            UserWarning,
            stacklevel=2,
        )

    B, H, N, D = query.shape

    if key.shape != (B, H, N, D):
        raise ValueError(
            f"Key shape {key.shape} doesn't match query {query.shape}. "
            "SageAttention3 requires identical Q/K/V shapes (no cross-attention or KV caching)."
        )
    if value.shape != (B, H, N, D):
        raise ValueError(
            f"Value shape {value.shape} doesn't match query {query.shape}. "
            "SageAttention3 requires identical Q/K/V shapes (no cross-attention or KV caching)."
        )

    use_per_block_mean = not _get_env_bool("SAGE3_DISABLE_PER_BLOCK_MEAN")
    quant_format = kwargs.pop("quant_format", _get_env_str("SAGE3_QUANT_FORMAT", "nvfp4"))
    acc_dtype = kwargs.pop("acc_dtype", _get_env_str("SAGE3_ACC_DTYPE", "fp32"))
    debug = _get_env_bool("SAGE3_DEBUG")

    if quant_format not in ATTENTION_CONFIGS:
        raise ValueError(f"Unknown quant_format '{quant_format}'. " f"Available: {list(ATTENTION_CONFIGS.keys())}")

    output = sageattn3_torch_triton_standalone(
        q=query,
        k=key,
        v=value,
        tensor_layout="HND",
        is_causal=is_causal,
        sm_scale=scale,
        per_block_mean=use_per_block_mean,
        tile_size_q=SAGE3_TILE_SIZE,
        tile_size_k=SAGE3_TILE_SIZE,
        debug=debug,
        quant_format=quant_format,
        acc_dtype=acc_dtype,
    )

    if debug and (torch.isnan(output).any() or torch.isinf(output).any()):
        raise RuntimeError("SageAttention3 produced NaN/Inf outputs")

    return output
