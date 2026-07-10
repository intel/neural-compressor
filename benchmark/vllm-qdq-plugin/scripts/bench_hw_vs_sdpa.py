"""Benchmark: SAGE3 HW kernels vs BF16 SDPA on [1, 40, 75600, 128]."""

import json
import os
import sys

import torch
import torch.nn.functional as F

# Resolve paths relative to this script to avoid hardcoded user-specific paths
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
# Bypass top-level vllm dependency by importing the sage3 subpackage directly
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "vllm_qdq_plugin", "sage3_attn"))
from sage3.api import sageattn3_standalone

# ── Config ──
B, H, N, D = 1, 40, 75600, 128
WARMUP = 3
ITERS = 10
DEVICE = "cuda"
DTYPE = torch.bfloat16


def bench(fn, warmup=WARMUP, iters=ITERS):
    """Returns mean latency in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def compute_attn_flops(B, H, N, D):
    """FLOPs for standard attention: 2*B*H*N*N*D (QK) + 2*B*H*N*N*D (PV) = 4*B*H*N^2*D."""
    return 4 * B * H * N * N * D


def main():
    print(f"Shape: [{B}, {H}, {N}, {D}] | dtype: {DTYPE} | warmup: {WARMUP} | iters: {ITERS}")
    print(f"Device: {torch.cuda.get_device_name()}")
    print("-" * 70)
    total_flops = compute_attn_flops(B, H, N, D)

    q = torch.randn(B, H, N, D, device=DEVICE, dtype=DTYPE)
    k = torch.randn(B, H, N, D, device=DEVICE, dtype=DTYPE)
    v = torch.randn(B, H, N, D, device=DEVICE, dtype=DTYPE)

    configs = [
        ("bf16_sdpa", lambda: F.scaled_dot_product_attention(q, k, v)),
        ("mxfp8_hw", lambda: sageattn3_standalone(q, k, v, config="mxfp8_hw")),
        ("mxfp4_hw", lambda: sageattn3_standalone(q, k, v, config="mxfp4_hw")),
        ("mixed_mxfp8qk_mxfp4pv_hw", lambda: sageattn3_standalone(q, k, v, config="mixed_mxfp8qk_mxfp4pv_hw")),
        ("mixed_mxfp4qk_mxfp8pv_hw", lambda: sageattn3_standalone(q, k, v, config="mixed_mxfp4qk_mxfp8pv_hw")),
    ]

    results = {}
    baseline_ms = None

    for name, fn in configs:
        try:
            # Verify it runs
            out = fn()
            assert out.shape == (B, H, N, D), f"Bad shape: {out.shape}"
            del out
            torch.cuda.empty_cache()

            ms = bench(fn)
            results[name] = {"ms": ms, "status": "ok"}
            if baseline_ms is None:
                baseline_ms = ms
        except Exception as e:
            results[name] = {"ms": None, "status": f"FAIL: {e}"}
            print(f"  {name}: FAILED - {e}")

    # Print table
    print(f"{'Config':<30} {'Latency (ms)':>12} {'TFLOPS':>8} {'Speedup':>8}")
    print("-" * 62)
    for name, r in results.items():
        if r["ms"] is not None:
            speedup = baseline_ms / r["ms"] if baseline_ms else 0
            tflops = total_flops / (r["ms"] * 1e-3) / 1e12
            print(f"{name:<30} {r['ms']:>12.2f} {tflops:>7.1f} {speedup:>7.2f}x")
            r["speedup"] = round(speedup, 3)
            r["tflops"] = round(tflops, 2)
        else:
            print(f"{name:<30} {'FAIL':>12} {'N/A':>8}")

    # Save raw results
    out_path = os.path.join(REPO_ROOT, "bench_hw_vs_sdpa_results.json")
    meta = {
        "shape": [B, H, N, D],
        "dtype": str(DTYPE),
        "warmup": WARMUP,
        "iters": ITERS,
        "device": torch.cuda.get_device_name(),
        "total_flops": total_flops,
    }
    with open(out_path, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    print(f"\nRaw results saved to: {out_path}")


if __name__ == "__main__":
    main()
