"""
Pre-quantization transforms: operations applied to Q, K, V before quantization.

Each transform has the signature:
    (q, k, v, ctx: TransformContext) → (q, k, v, ctx: TransformContext)

Transforms are composable via the pre_transforms list on AttentionConfig.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

# Type alias for transform functions
TransformFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, "TransformContext"],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, "TransformContext"],
]


@dataclass
class TransformContext:
    """
    Typed context passed between transforms and to the kernel launcher.

    Carries metadata produced by transforms that downstream stages need
    (e.g., delta_s from smoothing, v_mean from V-smoothing).
    """
    delta_s: Optional[torch.Tensor] = None
    v_mean: Optional[torch.Tensor] = None


# ============================================================================
# QK Smoothing
# ============================================================================

def _pad_128(x: torch.Tensor) -> torch.Tensor:
    """Pad tensor's sequence dimension to a multiple of 128."""
    L = x.size(2)
    pad_len = (128 - L % 128) % 128
    if pad_len == 0:
        return x.contiguous()
    return F.pad(x, (0, 0, 0, pad_len), value=0).contiguous()


def qk_smoothing(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ctx: TransformContext,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, TransformContext]:
    """
    Apply QK smoothing to reduce quantization outliers.

    1. K centering: k_centered = k - mean(k, dim=sequence)
    2. Q per-block smoothing: subtract per-128-token-group mean
    3. Delta correction: delta_s = q_means @ k_centered^T

    The delta_s correction is added back during attention to maintain equivalence.
    """
    B, H, N, D = q.shape

    # Step 1: K centering (lossless)
    k_centered = k - k.mean(dim=-2, keepdim=True)

    # Step 2: Pad to multiple of 128
    q_padded = _pad_128(q)
    k_padded = _pad_128(k_centered)

    # Step 3: Q smoothing with per-block means
    if N >= 128:
        L_pad = q_padded.size(2)
        GROUP_SIZE = 128
        num_groups = L_pad // GROUP_SIZE

        q_grouped = q_padded.view(B, H, num_groups, GROUP_SIZE, D)
        q_means = q_grouped.mean(dim=3, keepdim=False)  # [B, H, num_groups, D]
        q_smoothed_grouped = q_grouped - q_means.unsqueeze(3)
        q_smoothed_full = q_smoothed_grouped.view(B, H, L_pad, D)
    else:
        q_means = q_padded.mean(dim=-2, keepdim=True)  # [B, H, 1, D]
        q_smoothed_full = q_padded - q_means
        num_groups = 1

    # Remove padding
    q_smoothed = q_smoothed_full[:, :, :N, :]
    k_smoothed = k_padded[:, :, :N, :]

    # Step 4: Compute delta_s = q_means @ k^T
    ctx.delta_s = torch.matmul(
        q_means, k_smoothed.transpose(-2, -1)
    ).to(torch.float32).contiguous()

    return q_smoothed, k_smoothed, v, ctx


# ============================================================================
# V Smoothing
# ============================================================================

def v_smoothing(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ctx: TransformContext,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, TransformContext]:
    """
    Subtract V's mean to reduce outlier impact before quantization.

    The correction (output += v_mean) is applied post-kernel in api.py,
    reading ctx.v_mean. This works because sum(softmax_weights) = 1:
    output = sum(softmax * (v - v_mean)) + v_mean = sum(softmax * v).
    """
    ctx.v_mean = v.mean(dim=-2, keepdim=True)  # [B, H, 1, D]
    v_centered = v - ctx.v_mean
    return q, k, v_centered, ctx
