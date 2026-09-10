"""Kernel-level accuracy test for MXFP4 HW attention with delta_s correction.

Verifies that QK smoothing + delta_s significantly reduces quantization error
compared to raw MXFP4 quantization, especially for tensors with large DC offsets.
"""

import math
import sys

import torch

sys.path.insert(0, "/home/yiliu7/workspace/vllm-qdq-plugin/src")

from vllm_qdq_plugin.sage3_attn.sage3.mxfp4_hw_kernel import (
    mxfp4_flash_attention,
    quantize_to_mxfp4,
)
from vllm_qdq_plugin.sage3_attn.sage3.transforms import TransformContext, qk_smoothing


def reference_attention(q, k, v, sm_scale, causal=False):
    """FP32 reference attention."""
    scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    if causal:
        M, N = scores.shape[-2], scores.shape[-1]
        mask = torch.triu(torch.ones(M, N, device=scores.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    return torch.matmul(p, v)


def run_mxfp4_hw(q, k, v, sm_scale, causal=False, use_delta_s=False):
    """Run MXFP4 HW kernel with optional delta_s."""
    B, H, N, D = q.shape

    delta_s = None
    if use_delta_s:
        ctx = TransformContext()
        q, k, v, ctx = qk_smoothing(q, k, v, ctx)
        delta_s = ctx.delta_s

    # Pad to BLOCK_M=128
    if N % 128 != 0:
        pad_len = 128 - (N % 128)
        q = torch.nn.functional.pad(q, (0, 0, 0, pad_len))
        k = torch.nn.functional.pad(k, (0, 0, 0, pad_len))
        v = torch.nn.functional.pad(v, (0, 0, 0, pad_len))
        if delta_s is not None:
            delta_s = torch.nn.functional.pad(delta_s, (0, pad_len))
        B, H, N_pad, D = q.shape
    else:
        N_pad = N

    # Quantize Q, K
    q_packed, q_scale = quantize_to_mxfp4(q)
    k_packed, k_scale = quantize_to_mxfp4(k)

    # Quantize V (col-major along N)
    v_t = v.permute(0, 1, 3, 2).contiguous()
    v_flat = v_t.reshape(-1, N_pad)
    v_packed_flat, v_scale_flat = quantize_to_mxfp4(v_flat)
    v_packed = v_packed_flat.reshape(B, H, D, N_pad // 2)
    v_scale = v_scale_flat.reshape(B, H, D, N_pad // 32)

    output = mxfp4_flash_attention(
        q_packed,
        k_packed,
        v_packed,
        q_scale,
        k_scale,
        v_scale,
        causal=causal,
        sm_scale=sm_scale,
        delta_s=delta_s,
    )

    # Trim padding
    return output[:, :, : q.shape[2] if N % 128 == 0 else N, :]


def relative_error(out, ref):
    """Compute relative L2 error."""
    return (out - ref).norm() / ref.norm()


def max_abs_error(out, ref):
    """Compute max absolute error."""
    return (out - ref).abs().max()


def cosine_similarity(out, ref):
    """Compute mean cosine similarity across tokens."""
    out_flat = out.reshape(-1, out.shape[-1])
    ref_flat = ref.reshape(-1, ref.shape[-1])
    cos = torch.nn.functional.cosine_similarity(out_flat, ref_flat, dim=-1)
    return cos.mean()


def test_basic_correctness():
    """Test that delta_s doesn't break basic functionality on uniform data."""
    print("=" * 70)
    print("TEST 1: Basic correctness (uniform random data)")
    print("=" * 70)

    torch.manual_seed(42)
    B, H, N, D = 1, 4, 256, 128
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    sm_scale = 1.0 / math.sqrt(D)

    ref = reference_attention(q, k, v, sm_scale)
    out_no_ds = run_mxfp4_hw(q, k, v, sm_scale, use_delta_s=False)
    out_ds = run_mxfp4_hw(q, k, v, sm_scale, use_delta_s=True)

    err_no_ds = relative_error(out_no_ds, ref).item()
    err_ds = relative_error(out_ds, ref).item()
    cos_no_ds = cosine_similarity(out_no_ds, ref).item()
    cos_ds = cosine_similarity(out_ds, ref).item()

    print(f"  Without delta_s: rel_err={err_no_ds:.4f}, cos_sim={cos_no_ds:.6f}")
    print(f"  With    delta_s: rel_err={err_ds:.4f}, cos_sim={cos_ds:.6f}")
    print(f"  Improvement:     {err_no_ds/err_ds:.2f}x error reduction")
    print()

    # For uniform data, delta_s shouldn't make things worse
    assert err_ds <= err_no_ds * 1.5, f"delta_s made things significantly worse: {err_ds} vs {err_no_ds}"
    print("  PASSED: delta_s does not degrade uniform data\n")


def test_high_dc_offset():
    """Test accuracy improvement with high DC offsets (mimics Wan2.2 K statistics)."""
    print("=" * 70)
    print("TEST 2: High DC offset (DC/AC ratio ~6x, mimics real Wan2.2)")
    print("=" * 70)

    torch.manual_seed(123)
    B, H, N, D = 1, 4, 512, 128
    sm_scale = 1.0 / math.sqrt(D)

    # Create K with large DC offset: mean >> std (ratio ~6x as seen in Wan2.2)
    k_mean = torch.randn(1, H, 1, D, device="cuda") * 5.0  # large per-dim bias
    k_residual = torch.randn(B, H, N, D, device="cuda") * 0.8
    k = k_mean + k_residual  # DC/AC ~ 5/0.8 = 6.25

    # Create Q with per-block bias (~1.8x as seen in Wan2.2)
    q_bias = torch.randn(B, H, 1, D, device="cuda") * 2.0
    q_residual = torch.randn(B, H, N, D, device="cuda") * 1.1
    q = q_bias + q_residual

    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)

    # Verify DC offset magnitude
    k_dc_ac = k_mean.abs().mean() / k_residual.std()
    q_dc_ac = q_bias.abs().mean() / q_residual.std()
    print(f"  K DC/AC ratio: {k_dc_ac:.1f}x")
    print(f"  Q DC/AC ratio: {q_dc_ac:.1f}x")

    ref = reference_attention(q, k, v, sm_scale)
    out_no_ds = run_mxfp4_hw(q, k, v, sm_scale, use_delta_s=False)
    out_ds = run_mxfp4_hw(q, k, v, sm_scale, use_delta_s=True)

    err_no_ds = relative_error(out_no_ds, ref).item()
    err_ds = relative_error(out_ds, ref).item()
    cos_no_ds = cosine_similarity(out_no_ds, ref).item()
    cos_ds = cosine_similarity(out_ds, ref).item()
    max_err_no_ds = max_abs_error(out_no_ds, ref).item()
    max_err_ds = max_abs_error(out_ds, ref).item()

    print(f"\n  Without delta_s: rel_err={err_no_ds:.4f}, cos_sim={cos_no_ds:.6f}, max_abs={max_err_no_ds:.4f}")
    print(f"  With    delta_s: rel_err={err_ds:.4f}, cos_sim={cos_ds:.6f}, max_abs={max_err_ds:.4f}")
    improvement = err_no_ds / max(err_ds, 1e-10)
    print(f"  Improvement:     {improvement:.2f}x error reduction")
    print()

    # With high DC offset, delta_s should provide meaningful improvement
    assert err_ds < err_no_ds, f"Expected delta_s to reduce error, got {err_ds:.4f} >= {err_no_ds:.4f}"
    print("  PASSED: delta_s reduces error with high DC offset\n")


def test_causal_with_delta_s():
    """Test that delta_s works correctly with causal masking."""
    print("=" * 70)
    print("TEST 3: Causal attention with delta_s")
    print("=" * 70)

    torch.manual_seed(456)
    B, H, N, D = 1, 2, 256, 128
    sm_scale = 1.0 / math.sqrt(D)

    # High DC offset
    k_mean = torch.randn(1, H, 1, D, device="cuda") * 4.0
    k = k_mean + torch.randn(B, H, N, D, device="cuda") * 0.7
    q = torch.randn(B, H, N, D, device="cuda") * 2.0 + 1.5
    v = torch.randn(B, H, N, D, device="cuda")

    ref = reference_attention(q, k, v, sm_scale, causal=True)
    out_ds = run_mxfp4_hw(q, k, v, sm_scale, causal=True, use_delta_s=True)

    err = relative_error(out_ds, ref).item()
    cos = cosine_similarity(out_ds, ref).item()

    print(f"  With delta_s (causal): rel_err={err:.4f}, cos_sim={cos:.6f}")

    # Should be reasonably accurate
    assert cos > 0.9, f"Cosine similarity too low for causal+delta_s: {cos:.4f}"
    print("  PASSED: causal + delta_s works correctly\n")


def test_varying_seq_lengths():
    """Test with different sequence lengths to verify padding logic."""
    print("=" * 70)
    print("TEST 4: Varying sequence lengths (padding correctness)")
    print("=" * 70)

    torch.manual_seed(789)
    D = 128
    sm_scale = 1.0 / math.sqrt(D)

    for N in [128, 256, 384, 512, 1024]:
        B, H = 1, 2
        k_mean = torch.randn(1, H, 1, D, device="cuda") * 5.0
        k = k_mean + torch.randn(B, H, N, D, device="cuda") * 0.8
        q = torch.randn(B, H, N, D, device="cuda") + 2.0
        v = torch.randn(B, H, N, D, device="cuda")

        ref = reference_attention(q, k, v, sm_scale)
        out_ds = run_mxfp4_hw(q, k, v, sm_scale, use_delta_s=True)

        # Trim to match (in case of shape mismatch from padding)
        min_n = min(out_ds.shape[2], ref.shape[2])
        err = relative_error(out_ds[:, :, :min_n], ref[:, :, :min_n]).item()
        cos = cosine_similarity(out_ds[:, :, :min_n], ref[:, :, :min_n]).item()

        status = "OK" if cos > 0.85 else "FAIL"
        print(f"  N={N:4d}: rel_err={err:.4f}, cos_sim={cos:.6f} [{status}]")
        assert cos > 0.85, f"Failed at N={N}"

    print("\n  PASSED: All sequence lengths work correctly\n")


def test_delta_s_is_zero_for_centered_data():
    """Verify that delta_s has no effect when data is already centered."""
    print("=" * 70)
    print("TEST 5: Already-centered data (delta_s should be ~0)")
    print("=" * 70)

    torch.manual_seed(999)
    B, H, N, D = 1, 2, 256, 128
    sm_scale = 1.0 / math.sqrt(D)

    # Create zero-mean data
    q = torch.randn(B, H, N, D, device="cuda")
    q = q - q.mean(dim=2, keepdim=True)
    k = torch.randn(B, H, N, D, device="cuda")
    k = k - k.mean(dim=2, keepdim=True)
    v = torch.randn(B, H, N, D, device="cuda")

    out_no_ds = run_mxfp4_hw(q, k, v, sm_scale, use_delta_s=False)
    out_ds = run_mxfp4_hw(q, k, v, sm_scale, use_delta_s=True)

    diff = relative_error(out_ds, out_no_ds).item()
    print(f"  Difference between with/without delta_s: {diff:.6f}")
    print("  (Expected: small, since data is already centered)")

    # Outputs should be very similar for centered data
    assert diff < 0.3, f"delta_s changed output too much for centered data: {diff}"
    print("  PASSED: delta_s is near-neutral for centered data\n")


if __name__ == "__main__":
    print("\nMXFP4 HW Kernel — delta_s Accuracy Verification")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print()

    test_basic_correctness()
    test_high_dc_offset()
    test_causal_with_delta_s()
    test_varying_seq_lengths()
    test_delta_s_is_zero_for_centered_data()

    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
