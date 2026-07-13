# SPDX-License-Identifier: Apache-2.0
"""Verify the SpargeAttn backend integration: numeric accuracy + guard chain.

Run in the omni venv with SpargeAttn on the path:
    SPARGE_ATTN_REPO=/home/yiliu7/workspace/SpargeAttn \
    PYTHONPATH=/home/yiliu7/workspace/SpargeAttn \
    python tests/test_sparge_attn_integration.py

Checks:
  1. cos-sim >= 0.999 vs torch SDPA at topk=1.0 (dense) on REAL-structured input.
     (Random Q/K has no block structure, so topk<1.0 would legitimately tank
     cos-sim — we use topk=1.0 to isolate quantization error.)
  2. Guard chain: fp32 / seq_len<128 / head_dim=96 / attn_mask each take the SDPA
     path and never call the kernel.
"""

import sys

import torch
import torch.nn.functional as F


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


def _real_structured_qkv(B, N, H, D, device, dtype):
    """Build q/k/v with genuine attention structure (NHD = [B, N, H, D]).

    A few low-rank 'topic' directions shared between q and k create real
    block-level similarity, unlike pure noise. This mirrors what a trained
    attention layer produces and is the regime SpargeAttn is designed for.
    """
    g = torch.Generator(device=device).manual_seed(0)
    topics = torch.randn(H, 8, D, device=device, generator=g, dtype=torch.float32)
    # Per-token soft assignment to topics, smoothed along the sequence.
    qa = torch.randn(B, N, H, 8, device=device, generator=g, dtype=torch.float32)
    ka = torch.randn(B, N, H, 8, device=device, generator=g, dtype=torch.float32)
    qa = torch.cumsum(qa, dim=1) / (N**0.5)  # temporal smoothing -> structure
    ka = torch.cumsum(ka, dim=1) / (N**0.5)
    q = torch.einsum("bnht,htd->bnhd", qa.softmax(-1), topics)
    k = torch.einsum("bnht,htd->bnhd", ka.softmax(-1), topics)
    v = torch.randn(B, N, H, D, device=device, generator=g, dtype=torch.float32)
    return q.to(dtype), k.to(dtype), v.to(dtype)


def main() -> int:
    repo = None
    import os

    repo = os.environ.get("SPARGE_ATTN_REPO")
    if repo and repo not in sys.path:
        sys.path.insert(0, repo)

    # Import the impl (pulls in spas_sage_attn).
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from vllm_qdq_plugin.sparge_attn.impl import SpargeAttnImpl

    if not torch.cuda.is_available():
        print("FAIL: CUDA not available")
        return 1

    device = "cuda"
    dtype = torch.bfloat16
    B, N, H, D = 2, 1024, 16, 128
    sm_scale = 1.0 / (D**0.5)

    failures = []

    # --- Check 1: numeric accuracy at topk=1.0 (dense) on real-structured input ---
    q, k, v = _real_structured_qkv(B, N, H, D, device, dtype)
    impl = SpargeAttnImpl(num_heads=H, head_size=D, softmax_scale=sm_scale, causal=False)
    impl._mode = "topk"
    impl._topk = 1.0

    out_sparge = impl.forward_cuda(q, k, v)  # NHD in, NHD out
    # Reference SDPA in [B,H,N,D].
    ref = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        scale=sm_scale,
        is_causal=False,
    ).transpose(1, 2)
    cos = _cos(out_sparge, ref)
    ok1 = cos >= 0.999
    print(f"[1] dense topk=1.0 cos-sim vs SDPA: {cos:.5f}  ({'PASS' if ok1 else 'FAIL'})")
    if not ok1:
        failures.append(f"numeric cos-sim {cos:.5f} < 0.999")

    # --- Check 2: guard chain takes the SDPA path ---
    # Spy on _forward_sparge to assert it is NOT called on guarded inputs.
    def _make_impl():
        im = SpargeAttnImpl(num_heads=H, head_size=D, softmax_scale=sm_scale)
        im._sparge_called = False
        orig = im._forward_sparge

        def spy(*a, **kw):
            im._sparge_called = True
            return orig(*a, **kw)

        im._forward_sparge = spy
        return im

    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata

    cases = []
    # fp32
    im = _make_impl()
    q32, k32, v32 = (t.float() for t in _real_structured_qkv(B, N, H, D, device, torch.float32))
    im.forward_cuda(q32, k32, v32)
    cases.append(("fp32", im._sparge_called))
    # seq_len < 128
    im = _make_impl()
    qs, ks, vs = _real_structured_qkv(B, 64, H, D, device, dtype)
    im.forward_cuda(qs, ks, vs)
    cases.append(("seq_len<128", im._sparge_called))
    # head_dim = 96
    im = _make_impl()
    qd, kd, vd = _real_structured_qkv(B, N, H, 96, device, dtype)
    im.forward_cuda(qd, kd, vd)
    cases.append(("head_dim=96", im._sparge_called))
    # attn_mask present
    im = _make_impl()
    md = AttentionMetadata(attn_mask=torch.zeros(B, 1, N, N, device=device, dtype=dtype))
    im.forward_cuda(q, k, v, md)
    cases.append(("attn_mask", im._sparge_called))
    # cross-attention N_q != N_k
    im = _make_impl()
    qx, _, _ = _real_structured_qkv(B, N, H, D, device, dtype)
    _, kx, vx = _real_structured_qkv(B, 512, H, D, device, dtype)
    im.forward_cuda(qx, kx, vx)
    cases.append(("cross-attn", im._sparge_called))

    for name, called in cases:
        ok = not called
        print(f"[2] guard {name}: kernel called={called}  ({'PASS' if ok else 'FAIL'})")
        if not ok:
            failures.append(f"guard {name} did not fall back to SDPA")

    # --- Sanity: a valid input DOES call the kernel ---
    im = _make_impl()
    im.forward_cuda(q, k, v)
    ok_pos = im._sparge_called
    print(f"[3] valid input calls kernel: {im._sparge_called}  ({'PASS' if ok_pos else 'FAIL'})")
    if not ok_pos:
        failures.append("valid input did not reach the SpargeAttn kernel")

    print()
    if failures:
        print("RESULT: FAIL")
        for f in failures:
            print("  -", f)
        return 1
    print("RESULT: PASS — all checks green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
