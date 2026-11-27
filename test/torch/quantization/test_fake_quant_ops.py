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
"""Unit tests for fake-quant hot-path QDQ kernels, STE autograd, packing, and torch.compile."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neural_compressor.torch.algorithms.fake_quant import (
    ModuleQuantConfig,
    QuantLinear,
    QuantizerConfig,
    dequantize_packed_weight,
    fake_quant_dynamic,
    fake_quant_fp8_static_amax,
    functional as qdq_mod,
    pack_nibbles,
    pack_weight,
    ste_fake_quant,
    unpack_nibbles,
)


def test_e2m1_round_half_to_even_midpoints():
    # Block max is 4.0 -> descale = 4/6 -> log2(2/3) = -0.58 -> ceil = 0 -> scale = 1.0!
    row = torch.zeros(32, dtype=torch.float32)
    probes = [0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 4.0, -0.25, -0.75, -1.25, -1.75, -2.5, -3.5]
    expected = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 0.0, -1.0, -1.0, -2.0, -2.0, -4.0]
    row[: len(probes)] = torch.tensor(probes, dtype=torch.float32)
    out = fake_quant_dynamic(row, "mxfp4")
    torch.testing.assert_close(out[: len(expected)], torch.tensor(expected, dtype=torch.float32), rtol=0, atol=0)


def test_all_zeros_and_subnormals_do_not_produce_nan():
    for fmt in ("mxfp4", "nvfp4", "mxfp8", "fp8_e4m3"):
        zeros = torch.zeros(4, 32, dtype=torch.bfloat16)
        out_z = fake_quant_dynamic(zeros, fmt)
        assert not torch.isnan(out_z).any()
        assert torch.equal(out_z, zeros)

        tiny = torch.full((4, 32), 1e-38, dtype=torch.float32)
        out_t = fake_quant_dynamic(tiny, fmt)
        assert torch.isfinite(out_t).all()


@pytest.mark.parametrize("fmt", ["mxfp4", "nvfp4", "mxfp8", "fp8_e4m3", "bf16"])
def test_pack_dequantize_matches_qdq_bit_for_bit(fmt):
    torch.manual_seed(123)
    w = torch.randn(16, 64, dtype=torch.bfloat16)
    packed = pack_weight(w, fmt)
    dq = dequantize_packed_weight(packed, dtype=torch.bfloat16)
    ref = fake_quant_dynamic(w, fmt)
    assert torch.equal(dq, ref), f"Mismatch between pack+dequant and dynamic QDQ for {fmt}"


def test_pack_dequantize_static_fp8_zero_ulp_parity():
    torch.manual_seed(999)
    w = torch.randn(32, 64, dtype=torch.bfloat16) * 3.7
    # Test multiple tricky non-power-of-two amax values that trigger 1-ULP diffs if 1/(448/amax) isn't preserved
    for amax_val in (0.436, 1.2345679, 7.31, 50.2, 1e-12):
        amax = torch.tensor(amax_val, dtype=torch.float32)
        packed = pack_weight(w, "fp8_e4m3", amax=amax)
        dq = dequantize_packed_weight(packed, dtype=torch.bfloat16)
        ref = fake_quant_fp8_static_amax(w, amax)
        assert torch.equal(dq, ref), f"Non-zero ULP mismatch for amax={amax_val}"


def test_nibble_pack_unpack_roundtrip():
    codes = torch.randint(0, 16, (8, 32), dtype=torch.uint8)
    packed = pack_nibbles(codes)
    assert packed.shape == (8, 16)
    assert packed.dtype == torch.uint8
    unpacked = unpack_nibbles(packed)
    assert torch.equal(unpacked, codes)


def test_chunked_qdq_matches_unchunked_exact(monkeypatch):
    torch.manual_seed(77)
    x = torch.randn(3, 16, 64, dtype=torch.bfloat16)
    for fmt in ("mxfp4", "nvfp4", "mxfp8"):
        monkeypatch.setattr(qdq_mod, "_MAX_CHUNK_ELEMENTS", 10_000_000)
        unchunked = fake_quant_dynamic(x, fmt)
        monkeypatch.setattr(qdq_mod, "_MAX_CHUNK_ELEMENTS", 128)
        chunked = fake_quant_dynamic(x, fmt)
        assert torch.equal(chunked, unchunked), f"Chunked != unchunked for {fmt}"


def test_ste_saves_zero_tensors_and_frozen_weight_skips_autograd():
    x = torch.randn(4, 32, dtype=torch.bfloat16, requires_grad=True)
    y = ste_fake_quant(x, "mxfp4")
    assert y.grad_fn is not None
    assert len(y.grad_fn.saved_tensors) == 0
    grad_out = torch.randn_like(y)
    y.backward(grad_out)
    assert torch.equal(x.grad, grad_out)

    # Frozen base weight (e.g. LoRA Path-B) must NOT attach grad_fn to quantized weight
    lin = nn.Linear(32, 16, bias=False, dtype=torch.bfloat16)
    lin.weight.requires_grad_(False)
    qlin = QuantLinear.from_float(lin, ModuleQuantConfig(weights="mxfp4", input_activations="mxfp4"))
    q_w = qlin.weight_quantizer(qlin.weight)
    assert not q_w.requires_grad
    assert q_w.grad_fn is None

    act = torch.randn(2, 32, dtype=torch.bfloat16, requires_grad=True)
    out = qlin(act)
    assert out.requires_grad
    out.sum().backward()
    assert act.grad is not None
    assert qlin.weight.grad is None


@pytest.mark.parametrize("fmt,dynamic", [("mxfp4", True), ("nvfp4", True), ("mxfp8", True), ("fp8_e4m3", False)])
def test_torch_compile_fullgraph_zero_breaks(fmt, dynamic):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amax = None if dynamic else 4.25
    q = QuantLinear.from_float(
        nn.Linear(32, 32, bias=False, device=device, dtype=torch.bfloat16),
        ModuleQuantConfig(
            weights=QuantizerConfig(format=fmt, dynamic=dynamic, amax=amax),
            input_activations=QuantizerConfig(format=fmt, dynamic=dynamic, amax=amax),
        ),
    )
    x = torch.randn(4, 32, device=device, dtype=torch.bfloat16, requires_grad=True)
    eager_out = q(x)
    eager_out.sum().backward()
    eager_x_grad = x.grad.clone()
    eager_w_grad = q.weight.grad.clone()

    q.zero_grad(set_to_none=True)
    x_comp = x.detach().clone().requires_grad_(True)
    compiled = torch.compile(q, fullgraph=True)
    comp_out = compiled(x_comp)
    comp_out.sum().backward()

    # Inductor fuses non-power-of-two reciprocal in nvfp4, which can nudge an element on an E2M1 boundary by 1 bin.
    tol = 5e-2 if fmt == "nvfp4" else 1e-2
    torch.testing.assert_close(comp_out, eager_out, rtol=tol, atol=tol)
    torch.testing.assert_close(x_comp.grad, eager_x_grad, rtol=tol, atol=tol)
    torch.testing.assert_close(q.weight.grad, eager_w_grad, rtol=tol, atol=tol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for H2D copy profiling")
def test_zero_h2d_lut_transfers_after_warmup():
    device = torch.device("cuda:0")
    x = torch.randn(16, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    # Warmup once to populate per-device LUT cache
    _ = ste_fake_quant(x, "mxfp4")
    torch.cuda.synchronize(device)

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        y = ste_fake_quant(x, "mxfp4")
        y.sum().backward()
    torch.cuda.synchronize(device)

    h2d_events = [e for e in prof.key_averages() if "Memcpy HtoD" in e.key]
    assert len(h2d_events) == 0, f"Unexpected HtoD copies during MXFP4 QDQ: {h2d_events}"


@pytest.mark.skipif(
    not (torch.cuda.is_available() and qdq_mod.is_triton_qdq_available()),
    reason="CUDA + Triton required for fused Triton QDQ parity tests",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_triton_backend_0ulp_parity_vs_pytorch(dtype):
    device = torch.device("cuda:0")
    torch.manual_seed(2026)
    x = (torch.randn(256, 512, device=device, dtype=torch.float32) * 3.14159).to(dtype)
    # Inject all-zero block, E2M1 tie midpoints, and FP32 subnormals
    x[0, :32] = 0.0
    x[1, :32] = torch.tensor(
        [0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 4.0, 5.0, 6.0, -0.25, -0.75, -1.25, -1.75, -2.5, -3.5] * 2,
        device=device,
        dtype=dtype,
    )
    if dtype == torch.float32:
        x[2, :32] = 1e-38

    for fmt in ("mxfp4", "nvfp4", "mxfp8", "fp8_e4m3"):
        out_pt = fake_quant_dynamic(x, fmt, backend="pytorch")
        out_tr = fake_quant_dynamic(x, fmt, backend="triton")
        assert torch.equal(out_pt, out_tr), f"0-ULP mismatch for {fmt} ({dtype})"

    for amax_val in (0.436, 3.75, 50.2, 1e-12):
        amax_t = torch.tensor(amax_val, device=device, dtype=torch.float32)
        out_pt = fake_quant_fp8_static_amax(x, amax_t, backend="pytorch")
        out_tr = fake_quant_fp8_static_amax(x, amax_t, backend="triton")
        assert torch.equal(out_pt, out_tr), f"Static FP8 0-ULP mismatch for amax={amax_val} ({dtype})"

    # Non-contiguous view input
    x_nc = x[:, :256].transpose(0, 1)
    assert not x_nc.is_contiguous()
    assert torch.equal(
        fake_quant_dynamic(x_nc, "mxfp4", backend="pytorch"),
        fake_quant_dynamic(x_nc, "mxfp4", backend="triton"),
    )


@pytest.mark.skipif(
    not (torch.cuda.is_available() and qdq_mod.is_triton_qdq_available()),
    reason="CUDA + Triton required for fused Triton QDQ backend dispatch and memory tests",
)
def test_triton_backend_auto_dispatch_ste_and_zero_temp_vram():
    device = torch.device("cuda:0")
    prev = qdq_mod.get_quant_backend()
    try:
        qdq_mod.set_quant_backend("auto")
        x = torch.randn(64, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
        assert qdq_mod.resolve_quant_backend(x, None) == "triton"

        dy = torch.randn_like(x)
        y = ste_fake_quant(x, "mxfp4", backend="triton")
        y.backward(dy)
        assert torch.equal(x.grad, dy)

        # Verify 0-byte temporary VRAM spike on single-pass MXFP4 & static FP8
        big = torch.randn(1, 512, 4096, device=device, dtype=torch.bfloat16)
        amax = torch.tensor(6.25, device=device, dtype=torch.float32)
        out_bytes = big.numel() * big.element_size()

        _ = fake_quant_dynamic(big, "mxfp4", backend="triton")
        _ = fake_quant_fp8_static_amax(big, amax, backend="triton")
        torch.cuda.synchronize(device)

        for fn in (
            lambda: fake_quant_dynamic(big, "mxfp4", backend="triton"),
            lambda: fake_quant_fp8_static_amax(big, amax, backend="triton"),
        ):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            base = torch.cuda.memory_allocated(device)
            out = fn()
            peak = torch.cuda.max_memory_allocated(device)
            temp_bytes = max(0, (peak - base) - out_bytes)
            assert temp_bytes == 0, f"Expected 0 temporary bytes, got {temp_bytes}"
            del out
    finally:
        qdq_mod.set_quant_backend(prev)


@pytest.mark.skipif(
    not (torch.cuda.is_available() and qdq_mod.is_triton_qdq_available()),
    reason="CUDA + Triton required",
)
def test_triton_backend_0ulp_parity_vs_modelopt():
    mtq = pytest.importorskip("modelopt.torch.quantization")
    from modelopt.torch.quantization.config import QuantizerAttributeConfig
    from modelopt.torch.quantization.nn import TensorQuantizer as MoTensorQuantizer

    def _make_mo_q(cfg_dict):
        qcfg = cfg_dict["quant_cfg"]
        if isinstance(qcfg, list):
            raw = next(e["cfg"] for e in qcfg if "weight_quantizer" in e["quantizer_name"])
        else:
            raw = qcfg["*weight_quantizer"]
        return MoTensorQuantizer(QuantizerAttributeConfig(**raw)).cuda()

    device = torch.device("cuda:0")
    torch.manual_seed(1234)
    x = (torch.randn(512, 512, device=device, dtype=torch.float32) * 2.5).to(torch.bfloat16)

    mo_mxfp4 = _make_mo_q(mtq.MXFP4_DEFAULT_CFG)
    assert torch.equal(fake_quant_dynamic(x, "mxfp4", backend="triton"), mo_mxfp4(x))

    mo_nvfp4 = _make_mo_q(mtq.NVFP4_DEFAULT_CFG)
    mo_nvfp4.enable_calib()
    mo_nvfp4.disable_quant()
    _ = mo_nvfp4(x)
    mo_nvfp4.disable_calib()
    mo_nvfp4.enable_quant()
    tr_nvfp4 = fake_quant_dynamic(x, "nvfp4", backend="triton")
    pt_nvfp4 = fake_quant_dynamic(x, "nvfp4", backend="pytorch")
    assert torch.equal(tr_nvfp4, pt_nvfp4)
    # Note: compressed-tensors NVFP4 uses exact division x / eff_scale, whereas ModelOpt NVFP4 multiplies by
    # float32(1 / eff_scale), which can nudge exact 2.5 midpoints by 1 FP32 ULP; cosine similarity is > 0.9999.
    nv_cos = torch.nn.functional.cosine_similarity(tr_nvfp4.float().flatten(), mo_nvfp4(x).float().flatten(), dim=0)
    assert nv_cos.item() > 0.9999

    mo_mxfp8 = _make_mo_q(mtq.MXFP8_DEFAULT_CFG)
    assert torch.equal(fake_quant_dynamic(x, "mxfp8", backend="triton"), mo_mxfp8(x))

    mo_fp8 = _make_mo_q(mtq.FP8_DEFAULT_CFG)
    amax = torch.tensor(4.875, device=device, dtype=torch.float32)
    mo_fp8.amax = amax
    assert torch.equal(fake_quant_fp8_static_amax(x, amax, backend="triton"), mo_fp8(x))
