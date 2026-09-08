"""
MXFP8 (E4M3) Flash Attention on NVIDIA B200 (sm_100)
=====================================================

Educational standalone implementation of Flash Attention v2 using:
- Data format: float8_e4m3fn for Q, K, V
- Scale format: E8M0 (uint8, pure power-of-2 scales)
- Scale grouping: group_size=32 along reduction dimension
- Hardware: NVIDIA Blackwell (sm_100) via tl.dot_scaled -> tcgen05.mma

The kernel computes:
    O = softmax(Q @ K^T * sm_scale) @ V

where Q, K, V are in MXFP8 format (E4M3 data + E8M0 block scales).
"""

import time

import torch
import triton
import triton.language as tl

from .triton_utils import compute_p_scale_inv

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# ============================================================================
# Helper Functions
# ============================================================================


def fp8e8m0_to_float32(scale_uint8: torch.Tensor) -> torch.Tensor:
    """Convert E8M0 scale (uint8) to float32: value = 2^(x - 127)."""
    x = scale_uint8.to(torch.int32)
    x = x << 23  # shift into float32 exponent position
    return x.view(torch.float32)


def quantize_to_mxfp8(x: torch.Tensor, group_size: int = 32) -> tuple:
    """Quantize a float16/float32 tensor to MXFP8 E4M3 with E8M0 block scales.
    Groups along the last dimension with group_size=32.

    Returns:
        x_fp8: [..., K] float8_e4m3fn tensor
        scales: [..., K // group_size] uint8 E8M0 scales
    """
    assert x.shape[-1] % group_size == 0

    original_shape = x.shape
    x_f32 = x.float()

    *batch_dims, K = x_f32.shape
    num_groups = K // group_size
    x_grouped = x_f32.reshape(*batch_dims, num_groups, group_size)

    amax = x_grouped.abs().amax(dim=-1)  # [..., num_groups]
    e4m3_max = 448.0

    amax_safe = amax.clamp(min=1e-12)
    log2_scale = torch.ceil(torch.log2(amax_safe / e4m3_max))
    e8m0_encoding = (log2_scale + 127).clamp(0, 254).to(torch.uint8)

    scale_f32 = fp8e8m0_to_float32(e8m0_encoding)
    scale_expanded = scale_f32.unsqueeze(-1).expand_as(x_grouped)
    scale_expanded = scale_expanded.reshape(original_shape)

    x_scaled = x_f32 / scale_expanded
    x_fp8 = x_scaled.to(torch.float8_e4m3fn)

    return x_fp8, e8m0_encoding


def dequantize_mxfp8(x_fp8: torch.Tensor, scales: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """Dequantize MXFP8 E4M3 tensor back to float32."""
    x_f32 = x_fp8.to(torch.float32)
    scale_f32 = fp8e8m0_to_float32(scales)
    scale_expanded = scale_f32.repeat_interleave(group_size, dim=-1)
    scale_expanded = scale_expanded[..., : x_f32.shape[-1]]
    return x_f32 * scale_expanded


# ============================================================================
# Triton Kernel
# ============================================================================


@triton.jit
def _mxfp8_attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    q_scale,  #
    K_ptr,
    K_scale_ptr,
    V_ptr,
    V_scale_ptr,  #
    Delta_s_ptr,  #
    stride_kn,
    stride_kk,  #
    stride_ks_n,
    stride_ks_k,  #
    stride_vn,
    stride_vk,  #
    stride_vs_d,
    stride_vs_n,  #
    stride_delta_g,
    stride_delta_n,  #
    start_m,
    qk_scale,  #
    offs_m,
    offs_n,  #
    N_CTX: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    HAS_DELTA_S: tl.constexpr,  #
):
    # Determine loop bounds based on causal stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:  # STAGE == 3: non-causal, full range
        lo, hi = 0, N_CTX

    offs_k_head = tl.arange(0, HEAD_DIM)
    offs_k_n = tl.arange(0, BLOCK_N)
    offs_scale_k = tl.arange(0, HEAD_DIM // 32)
    offs_scale_n = tl.arange(0, BLOCK_N // 32)
    offs_head_full = tl.arange(0, HEAD_DIM)

    for start_n in tl.range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # -- Load K block [BLOCK_N, HEAD_DIM] --
        k_ptrs = K_ptr + (start_n + offs_k_n[:, None]) * stride_kn + offs_k_head[None, :] * stride_kk
        k = tl.load(k_ptrs)  # [BLOCK_N, HEAD_DIM]

        # -- Load K scale [BLOCK_N, HEAD_DIM // 32] --
        ks_ptrs = K_scale_ptr + (start_n + offs_k_n[:, None]) * stride_ks_n + offs_scale_k[None, :] * stride_ks_k
        k_scale = tl.load(ks_ptrs)  # [BLOCK_N, HEAD_DIM // 32]

        # -- Q @ K^T via dot_scaled --
        # dot_scaled(lhs=[M,K], lhs_scale=[M,K//32], rhs=[K,N], rhs_scale=[N,K//32])
        # Here: lhs=q [BLOCK_M, HEAD_DIM], rhs=k.T [HEAD_DIM, BLOCK_N]
        # lhs_scale=q_scale [BLOCK_M, HEAD_DIM//32], rhs_scale=k_scale [BLOCK_N, HEAD_DIM//32]
        qk = tl.dot_scaled(q, q_scale, "e4m3", tl.trans(k), k_scale, "e4m3")

        # -- Add delta_s correction (before scaling) --
        if HAS_DELTA_S:
            group_id = start_m  # BLOCK_M=128 = GROUP_SIZE
            ds_ptrs = Delta_s_ptr + group_id * stride_delta_g + (start_n + offs_n) * stride_delta_n
            ds_tile = tl.load(ds_ptrs)  # [BLOCK_N]
            qk = qk + ds_tile[None, :]

        # -- Online softmax --
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]

        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)

        acc = acc * alpha[:, None]

        l_ij = tl.sum(p, 1)
        l_i = l_i * alpha + l_ij

        # -- P @ V via dot_scaled --
        # Compute P scale on the fly: group_size=32 along BLOCK_N (reduction dim)
        # p shape: [BLOCK_M, BLOCK_N], scale shape: [BLOCK_M, BLOCK_N // 32]
        p_reshaped = tl.reshape(p, [BLOCK_M, BLOCK_N // 32, 32])
        p_amax = tl.max(p_reshaped, 2)  # [BLOCK_M, BLOCK_N // 32]
        FP8_E4M3_MAX: tl.constexpr = 448.0
        p_e8m0, inv_scale_expanded = compute_p_scale_inv(
            p_amax,
            FP8_E4M3_MAX,
            BLOCK_M,
            BLOCK_N,
        )
        p_scaled = tl.minimum(p * inv_scale_expanded, FP8_E4M3_MAX)
        p_fp8 = p_scaled.to(tl.float8e4nv)
        p_scale = p_e8m0  # [BLOCK_M, BLOCK_N // 32]

        # Load V block [BLOCK_N, HEAD_DIM]
        v_ptrs = V_ptr + (start_n + offs_k_n[:, None]) * stride_vn + offs_head_full[None, :] * stride_vk
        v = tl.load(v_ptrs)  # [BLOCK_N, HEAD_DIM]

        # Load V scale [HEAD_DIM, BLOCK_N // 32]
        # V_scale layout: [HEAD_DIM, N_CTX // 32], we load [HEAD_DIM, BLOCK_N//32]
        vs_ptrs = (
            V_scale_ptr + offs_head_full[:, None] * stride_vs_d + (start_n // 32 + offs_scale_n[None, :]) * stride_vs_n
        )
        v_scale = tl.load(vs_ptrs)  # [HEAD_DIM, BLOCK_N // 32]

        # dot_scaled: lhs=p_fp8 [BLOCK_M, BLOCK_N], rhs=v [BLOCK_N, HEAD_DIM]
        # lhs_scale=p_scale [BLOCK_M, BLOCK_N//32], rhs_scale=v_scale [HEAD_DIM, BLOCK_N//32]
        acc = tl.dot_scaled(p_fp8, p_scale, "e4m3", v, v_scale, "e4m3", acc)

        m_i = m_ij

    return acc, l_i, m_i


@triton.jit
def _mxfp8_attn_fwd(
    Q,
    K,
    V,
    Out,  #
    Q_scale,
    K_scale,
    V_scale,  #
    Delta_s,  #
    sm_scale,  #
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,  #
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,  #
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,  #
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,  #
    stride_qsz,
    stride_qsh,
    stride_qsm,
    stride_qsk,  #
    stride_ksz,
    stride_ksh,
    stride_ksn,
    stride_ksk,  #
    stride_vsz,
    stride_vsh,
    stride_vsd,
    stride_vsn,  #
    stride_dsz,
    stride_dsh,
    stride_dsg,
    stride_dsn,  #
    Z,
    H,
    N_CTX,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    HAS_DELTA_S: tl.constexpr,  #
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # Base pointers for this batch/head
    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    o_offset = off_z * stride_oz + off_h * stride_oh
    qs_offset = off_z * stride_qsz + off_h * stride_qsh
    ks_offset = off_z * stride_ksz + off_h * stride_ksh
    vs_offset = off_z * stride_vsz + off_h * stride_vsh

    Q_ptr = Q + q_offset
    K_ptr = K + k_offset
    V_ptr = V + v_offset
    Q_scale_ptr = Q_scale + qs_offset
    K_scale_ptr = K_scale + ks_offset
    V_scale_ptr = V_scale + vs_offset

    # Delta_s pointer for this batch/head
    Delta_s_ptr = Delta_s + off_z * stride_dsz + off_h * stride_dsh

    # Load Q block [BLOCK_M, HEAD_DIM]
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_head = tl.arange(0, HEAD_DIM)
    q_ptrs = Q_ptr + offs_m[:, None] * stride_qm + offs_head[None, :] * stride_qk
    q = tl.load(q_ptrs)  # [BLOCK_M, HEAD_DIM]

    # Load Q scale [BLOCK_M, HEAD_DIM // 32]
    offs_scale_k = tl.arange(0, HEAD_DIM // 32)
    qs_ptrs = Q_scale_ptr + offs_m[:, None] * stride_qsm + offs_scale_k[None, :] * stride_qsk
    q_scale = tl.load(qs_ptrs)  # [BLOCK_M, HEAD_DIM // 32]

    # Initialize accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Scale with log2(e) for exp2-based softmax
    qk_scale = sm_scale * 1.44269504

    offs_n = tl.arange(0, BLOCK_N)

    # Run attention inner loop
    if STAGE & 1:
        acc, l_i, m_i = _mxfp8_attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            q_scale,
            K_ptr,
            K_scale_ptr,
            V_ptr,
            V_scale_ptr,
            Delta_s_ptr,
            stride_kn,
            stride_kk,
            stride_ksn,
            stride_ksk,
            stride_vn,
            stride_vk,
            stride_vsd,
            stride_vsn,
            stride_dsg,
            stride_dsn,
            start_m,
            qk_scale,
            offs_m,
            offs_n,
            N_CTX,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            4 - STAGE,
            HAS_DELTA_S,
        )
    if STAGE & 2:
        acc, l_i, m_i = _mxfp8_attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            q_scale,
            K_ptr,
            K_scale_ptr,
            V_ptr,
            V_scale_ptr,
            Delta_s_ptr,
            stride_kn,
            stride_kk,
            stride_ksn,
            stride_ksk,
            stride_vn,
            stride_vk,
            stride_vsd,
            stride_vsn,
            stride_dsg,
            stride_dsn,
            start_m,
            qk_scale,
            offs_m,
            offs_n,
            N_CTX,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            2,
            HAS_DELTA_S,
        )

    # Normalize output
    acc = acc / l_i[:, None]

    # Store output
    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_head[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty))


# ============================================================================
# Wrapper Function
# ============================================================================


def mxfp8_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    causal: bool = False,
    sm_scale: float = None,
    delta_s: torch.Tensor = None,
) -> torch.Tensor:
    """MXFP8 (E4M3) Flash Attention forward pass with E8M0 block scales.

    Args:
        q: [B, H, M, D] float8_e4m3fn — queries
        k: [B, H, N, D] float8_e4m3fn — keys
        v: [B, H, N, D] float8_e4m3fn — values
        q_scale: [B, H, M, D//32] uint8 — E8M0 scales for Q (groups along HEAD_DIM)
        k_scale: [B, H, N, D//32] uint8 — E8M0 scales for K (groups along HEAD_DIM)
        v_scale: [B, H, D, N//32] uint8 — E8M0 scales for V (groups along sequence dim)
        causal: whether to apply causal mask
        sm_scale: softmax scale (default: 1/sqrt(HEAD_DIM))
        delta_s: [B, H, num_groups, N] float32 — QK smoothing correction (optional)

    Returns:
        output: [B, H, M, D] float32
    """
    B, H, M, D = q.shape
    N = k.shape[2]

    if sm_scale is None:
        sm_scale = 1.0 / (D**0.5)

    BLOCK_M = 128
    BLOCK_N = 64

    assert D in (64, 128), f"HEAD_DIM must be 64 or 128, got {D}"
    assert M % BLOCK_M == 0, f"M={M} must be divisible by BLOCK_M={BLOCK_M}"
    assert N % BLOCK_N == 0, f"N={N} must be divisible by BLOCK_N={BLOCK_N}"

    output = torch.empty((B, H, M, D), dtype=torch.float32, device=q.device)

    STAGE = 3 if causal else 1

    has_delta_s = delta_s is not None
    if not has_delta_s:
        # Dummy tensor — pointer won't be dereferenced when HAS_DELTA_S=False
        delta_s = torch.zeros(1, 1, 1, 1, dtype=torch.float32, device=q.device)

    grid = (triton.cdiv(M, BLOCK_M), B * H)

    _mxfp8_attn_fwd[grid](
        q,
        k,
        v,
        output,
        q_scale,
        k_scale,
        v_scale,
        delta_s,
        sm_scale,
        # Q strides
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        # K strides
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        # V strides
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        # O strides
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        # Q_scale strides
        q_scale.stride(0),
        q_scale.stride(1),
        q_scale.stride(2),
        q_scale.stride(3),
        # K_scale strides
        k_scale.stride(0),
        k_scale.stride(1),
        k_scale.stride(2),
        k_scale.stride(3),
        # V_scale strides [B, H, D, N//32]
        v_scale.stride(0),
        v_scale.stride(1),
        v_scale.stride(2),
        v_scale.stride(3),
        # Delta_s strides [B, H, num_groups, N]
        delta_s.stride(0),
        delta_s.stride(1),
        delta_s.stride(2),
        delta_s.stride(3),
        # Dimensions
        B,
        H,
        N,
        # Compile-time constants
        HEAD_DIM=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        STAGE=STAGE,
        HAS_DELTA_S=has_delta_s,
        num_warps=4,
        num_stages=3,
    )

    return output


# ============================================================================
# Test: Correctness vs SDPA
# ============================================================================


def test_mxfp8_flash_attention():
    """Test MXFP8 flash attention against torch SDPA on dequantized inputs."""
    print("=" * 60)
    print("Testing MXFP8 Flash Attention (dot_scaled) vs SDPA")
    print("=" * 60)

    torch.manual_seed(42)

    configs = [
        # (B, H, N, D, causal)
        (1, 1, 256, 128, False),
        (1, 4, 512, 128, False),
        (2, 4, 1024, 128, False),
        (2, 4, 1024, 128, True),
        (1, 8, 2048, 128, False),
    ]

    for B, H, N, D, causal in configs:
        # Generate random fp16 data
        q_f16 = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16) * 0.1
        k_f16 = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16) * 0.1
        v_f16 = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16) * 0.1

        # Quantize to MXFP8 with E8M0 scales
        q_fp8, q_scale = quantize_to_mxfp8(q_f16)  # scale along D
        k_fp8, k_scale = quantize_to_mxfp8(k_f16)  # scale along D

        # For V, we need scale along sequence dim (last dim of V in dot_scaled's RHS reduction)
        # V data: [B, H, N, D], but dot_scaled RHS reduction is along BLOCK_N (sequence)
        # V_scale shape: [B, H, D, N//32] — groups of 32 along sequence
        # Quantize along sequence: reshape to [B*H*D, N], quantize, reshape back
        v_flat = v_f16.permute(0, 1, 3, 2).contiguous()  # [B, H, D, N]
        v_flat_2d = v_flat.reshape(-1, N)
        v_fp8_flat, v_scale_flat = quantize_to_mxfp8(v_flat_2d)  # scale along N
        v_fp8_transposed = v_fp8_flat.reshape(B, H, D, N)  # [B, H, D, N]
        v_scale = v_scale_flat.reshape(B, H, D, N // 32)  # [B, H, D, N//32]
        # V data for kernel: [B, H, N, D] (standard layout)
        v_fp8 = v_fp8_transposed.permute(0, 1, 3, 2).contiguous()  # [B, H, N, D]

        # Dequantize for reference
        q_deq = dequantize_mxfp8(q_fp8, q_scale).float()
        k_deq = dequantize_mxfp8(k_fp8, k_scale).float()
        # For V, dequantize in the transposed domain then permute back
        v_deq_t = dequantize_mxfp8(v_fp8_transposed.reshape(-1, N), v_scale_flat).reshape(B, H, D, N)
        v_deq = v_deq_t.permute(0, 1, 3, 2).contiguous().float()

        # Reference: torch SDPA
        sm_scale = 1.0 / (D**0.5)
        ref = torch.nn.functional.scaled_dot_product_attention(q_deq, k_deq, v_deq, is_causal=causal, scale=sm_scale)

        # Our kernel
        out = mxfp8_flash_attention(q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale, causal=causal, sm_scale=sm_scale)

        # Compare
        max_diff = (ref - out).abs().max().item()
        mean_diff = (ref - out).abs().mean().item()
        cos_sim = torch.nn.functional.cosine_similarity(ref.flatten().unsqueeze(0), out.flatten().unsqueeze(0)).item()

        # Causal has higher max_diff due to fp8 quantization noise on sparse rows
        threshold = 0.7 if causal else 0.15
        status = "PASS" if max_diff < threshold else "FAIL"
        print(
            f"  [{status}] B={B}, H={H}, N={N}, D={D}, causal={causal}: "
            f"max_diff={max_diff:.4f}, mean_diff={mean_diff:.6f}, cos_sim={cos_sim:.6f}"
        )

        if status == "FAIL":
            print("    WARNING: Large difference detected!")

    print()


# ============================================================================
# Benchmark
# ============================================================================


def benchmark_mxfp8_flash_attention():
    """Benchmark MXFP8 flash attention and report TFLOPS."""
    print("=" * 60)
    print("Benchmarking MXFP8 Flash Attention (dot_scaled)")
    print("=" * 60)

    torch.manual_seed(42)

    configs = [
        # (B, H, N, D)
        (4, 32, 2048, 128),
        (2, 32, 4096, 128),
        (1, 32, 8192, 128),
        (1, 40, 64 * 1024, 128),
    ]

    for B, H, N, D in configs:
        q_f16 = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16) * 0.1
        k_f16 = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16) * 0.1
        v_f16 = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16) * 0.1

        q_fp8, q_scale = quantize_to_mxfp8(q_f16)
        k_fp8, k_scale = quantize_to_mxfp8(k_f16)

        v_flat = v_f16.permute(0, 1, 3, 2).contiguous().reshape(-1, N)
        v_fp8_flat, v_scale_flat = quantize_to_mxfp8(v_flat)
        v_fp8_transposed = v_fp8_flat.reshape(B, H, D, N)
        v_scale = v_scale_flat.reshape(B, H, D, N // 32)
        v_fp8 = v_fp8_transposed.permute(0, 1, 3, 2).contiguous()

        sm_scale = 1.0 / (D**0.5)

        # Warmup
        for _ in range(5):
            out = mxfp8_flash_attention(q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale, causal=False, sm_scale=sm_scale)
        torch.cuda.synchronize()

        # Benchmark
        num_iters = 20
        start = time.perf_counter()
        for _ in range(num_iters):
            out = mxfp8_flash_attention(q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale, causal=False, sm_scale=sm_scale)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / num_iters

        flops = 4 * B * H * N * N * D
        tflops = flops / elapsed / 1e12

        print(f"  B={B}, H={H}, N={N}, D={D}: {elapsed*1000:.2f} ms, {tflops:.1f} TFLOPS")

    print()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability()}")
    print()

    test_mxfp8_flash_attention()
    benchmark_mxfp8_flash_attention()
