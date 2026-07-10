"""
Host-side K/V quantization: fake-quantize Q, K, V tensors before feeding to the attention kernel.

Q/K are blocked along D (head_dim), V is blocked along N (seq_len).
The block size, fp_max, and rounding functions come from QuantConfig.
"""

import torch
import torch.nn.functional as F
from typing import Tuple

from .quant_config import QuantConfig
from .quant_primitives import apply_e2m1_quantization_torch

# Epsilon to prevent division by zero in scale computation.
# Applied both pre- and post-rounding because E4M3 rounding can map
# small positive values (< ~1.95e-3) to zero.
SCALE_EPSILON = 1e-8


def quantize_qk(x: torch.Tensor, config: QuantConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-block microscaling quantization for Q or K tensors.

    Blocks are formed along the D (head_dim) dimension.

    Steps:
        1. Reshape tensor into blocks of config.block_size along D
        2. Per-block max → scale = block_max / fp_max, rounded via config
        3. Normalize, quantize, reconstruct

    Args:
        x: Input tensor [B, H, N, D]
        config: QuantConfig with block_size, fp_max, round_scale_torch, quant_data_torch

    Returns:
        (x_quantized [B, H, N, D], scales [B, H, N, D//block_size])
    """
    B, H, N, D = x.shape
    block_size = config.block_size

    # Handle dimensions not divisible by block_size
    if D % block_size != 0:
        pad_size = block_size - (D % block_size)
        x_padded = F.pad(x, (0, pad_size), mode='constant', value=0)
        D_padded = D + pad_size
    else:
        x_padded = x
        D_padded = D

    # Reshape for D-dimension aligned microscaling blocks
    num_blocks = D_padded // block_size
    x_blocks = x_padded.view(B, H, N, num_blocks, block_size)

    # Per-block scaling
    block_max = x_blocks.abs().max(dim=-1)[0]  # [B, H, N, num_blocks]
    scales = block_max / config.fp_max
    scales = torch.clamp(scales, min=SCALE_EPSILON)
    scales = config.round_scale_torch(scales)
    scales = torch.clamp(scales, min=SCALE_EPSILON)  # E4M3 can round small values to 0

    # Normalize to quantization range
    x_normalized = x_blocks / scales.unsqueeze(-1)

    # Apply data quantization
    _data_quant = config.quant_data_torch or apply_e2m1_quantization_torch
    x_quantized_blocks = _data_quant(x_normalized)

    # Reconstruct with scales
    x_quantized_blocks = x_quantized_blocks * scales.unsqueeze(-1)

    # Flatten back
    x_quantized_full = x_quantized_blocks.view(B, H, N, D_padded)

    # Remove padding if added
    if D_padded != D:
        x_quantized = x_quantized_full[:, :, :, :D].contiguous()
    else:
        x_quantized = x_quantized_full

    return x_quantized, scales


def quantize_v(x: torch.Tensor, config: QuantConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-block microscaling quantization for V tensor.

    Unlike Q/K which block along D (head_dim), V is quantized with blocks
    along N (seq_len).

    Args:
        x: Input V tensor [B, H, N, D]
        config: QuantConfig with block_size, fp_max, round_scale_torch, quant_data_torch

    Returns:
        (x_quantized [B, H, N, D], scales [B, H, D, N//block_size])
    """
    B, H, N, D = x.shape
    block_size = config.block_size

    # Pad N to multiple of block_size
    if N % block_size != 0:
        pad_size = block_size - (N % block_size)
        x_padded = F.pad(x, (0, 0, 0, pad_size), mode='constant', value=0)
        N_padded = N + pad_size
    else:
        x_padded = x
        N_padded = N

    # V blocks along N: transpose to [B, H, D, N], then block along last dim
    x_transposed = x_padded.transpose(-1, -2).contiguous()  # [B, H, D, N_padded]
    num_blocks = N_padded // block_size
    x_blocks = x_transposed.view(B, H, D, num_blocks, block_size)

    # Per-block scaling
    block_max = x_blocks.abs().max(dim=-1)[0]  # [B, H, D, num_blocks]
    scales = block_max / config.fp_max
    scales = torch.clamp(scales, min=SCALE_EPSILON)
    scales = config.round_scale_torch(scales)
    scales = torch.clamp(scales, min=SCALE_EPSILON)  # E4M3 can round small values to 0

    # Normalize and quantize
    x_normalized = x_blocks / scales.unsqueeze(-1)

    _data_quant = config.quant_data_torch or apply_e2m1_quantization_torch
    x_quantized_blocks = _data_quant(x_normalized)

    # Reconstruct with scales
    x_quantized_blocks = x_quantized_blocks * scales.unsqueeze(-1)

    # Flatten back to [B, H, D, N_padded] and transpose to [B, H, N_padded, D]
    x_quantized_transposed = x_quantized_blocks.view(B, H, D, N_padded)
    x_quantized_full = x_quantized_transposed.transpose(-1, -2).contiguous()

    # Remove padding if added
    if N_padded != N:
        x_quantized = x_quantized_full[:, :, :N, :].contiguous()
    else:
        x_quantized = x_quantized_full

    return x_quantized, scales
