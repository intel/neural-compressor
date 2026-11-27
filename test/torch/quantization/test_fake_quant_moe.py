# Copyright (c) 2026 Intel Corporation
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
"""Unit tests for 3D Fused MoE fake-quantization (`QuantFusedMoEExperts`), 3D LoRA, and CT export."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from neural_compressor.torch.algorithms.fake_quant import (
    FakeQuantConfig,
    QuantFusedMoEExperts,
    QuantLinear,
    attach_moe_expert_lora,
    get_quantizer_state_dict,
    load_quantizer_state_dict,
    merge_lora_into_quant_linear,
)
from neural_compressor.torch.distillation.trainer import _disable_quantizers, _self_teacher_context
from neural_compressor.torch.export.compressed_tensors import (
    dequantize_compressed_tensors_layer,
    pack_model_to_compressed_tensors,
)
from neural_compressor.torch.qad import prepare_qad_model
from neural_compressor.torch.quantization import prepare_qat


class DummyQwen3MoeExperts(nn.Module):
    """Minimal HF-compatible 3D fused SwiGLU MoE experts container (`Qwen3MoeExperts` signature)."""

    def __init__(self, num_experts: int = 4, hidden_dim: int = 64, intermediate_dim: int = 32) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.gate_up_proj = nn.Parameter(torch.randn(num_experts, 2 * intermediate_dim, hidden_dim) * 0.02)
        self.down_proj = nn.Parameter(torch.randn(num_experts, hidden_dim, intermediate_dim) * 0.02)
        self.act_fn = nn.SiLU()

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden = torch.zeros_like(hidden_states)
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        for e in range(self.num_experts):
            top_k_pos, token_idx = torch.where(expert_mask[e])
            if token_idx.numel() == 0:
                continue
            cur = hidden_states[token_idx]
            gate, up = nn.functional.linear(cur, self.gate_up_proj[e]).chunk(2, dim=-1)
            cur = nn.functional.linear(self.act_fn(gate) * up, self.down_proj[e])
            cur = cur * top_k_weights[token_idx, top_k_pos, None]
            final_hidden.index_add_(0, token_idx, cur.to(final_hidden.dtype))
        return final_hidden


class TinyMoEDecoderModel(nn.Module):
    """Tiny decoder block mimicking `Qwen3.5-MoE` (router gate + 3D experts + shared_expert + shared_expert_gate)."""

    def __init__(self, hidden_dim: int = 64, intermediate_dim: int = 32, num_experts: int = 4, top_k: int = 2) -> None:
        super().__init__()
        self.top_k = top_k
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.self_attn.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.mlp = nn.Module()
        self.mlp.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.mlp.experts = DummyQwen3MoeExperts(num_experts, hidden_dim, intermediate_dim)
        self.mlp.shared_expert_gate = nn.Linear(hidden_dim, 1, bias=False)
        self.mlp.shared_expert = nn.Module()
        self.mlp.shared_expert.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.mlp.shared_expert.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.mlp.shared_expert.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.lm_head = nn.Linear(hidden_dim, 48, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, dim = x.shape
        h = self.self_attn.o_proj(self.self_attn.q_proj(x))
        flat = h.reshape(-1, dim)
        logits = torch.softmax(self.mlp.gate(flat), dim=-1, dtype=torch.float32)
        top_w, top_idx = torch.topk(logits, self.top_k, dim=-1)
        top_w = (top_w / top_w.sum(dim=-1, keepdim=True)).to(flat.dtype)
        routed = self.mlp.experts(flat, top_idx, top_w)
        sh = self.mlp.shared_expert.down_proj(
            torch.nn.functional.silu(self.mlp.shared_expert.gate_proj(flat)) * self.mlp.shared_expert.up_proj(flat)
        )
        sh = torch.sigmoid(self.mlp.shared_expert_gate(flat)) * sh
        out = h + (routed + sh).reshape(bsz, seq, dim)
        return self.lm_head(out)


def test_qwen35_hybrid_converts_3d_experts_and_skips_routers():
    torch.manual_seed(0)
    model = TinyMoEDecoderModel()
    ref = copy.deepcopy(model)
    cfg = FakeQuantConfig.qwen3_5_mxfp4_hybrid(backend="pytorch")
    prepare_qat(model, cfg)

    assert isinstance(model.self_attn.q_proj, QuantLinear)
    assert isinstance(model.self_attn.o_proj, QuantLinear)
    assert isinstance(model.mlp.experts, QuantFusedMoEExperts)
    assert isinstance(model.mlp.shared_expert.gate_proj, QuantLinear)
    assert isinstance(model.mlp.shared_expert.down_proj, QuantLinear)
    assert type(model.mlp.gate) is nn.Linear
    assert type(model.mlp.shared_expert_gate) is nn.Linear
    assert type(model.lm_head) is nn.Linear

    x = torch.randn(2, 8, 64)
    with _disable_quantizers(model):
        y_disabled = model(x)
    y_ref = ref(x)
    assert torch.allclose(y_disabled, y_ref, atol=1e-5, rtol=1e-5)

    for m in model.modules():
        for attr in ("weight_quantizer", "input_quantizer"):
            q = getattr(m, attr, None)
            if q is not None and q.is_enabled and not q.is_dynamic and q.amax is None:
                q.set_amax(1.0)

    y_quant = model(x)
    y_quant.sum().backward()
    assert model.mlp.experts.gate_up_proj.grad is not None
    assert torch.isfinite(model.mlp.experts.gate_up_proj.grad).all()
    assert model.mlp.experts.down_proj.grad is not None
    assert torch.isfinite(model.mlp.experts.down_proj.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for BF16 F.grouped_mm path")
def test_grouped_mm_cuda_bf16_forward_backward_and_3d_lora_self_teacher():
    torch.manual_seed(7)
    device = torch.device("cuda:0")
    model = TinyMoEDecoderModel(hidden_dim=128, intermediate_dim=64, num_experts=6, top_k=2).to(
        device=device, dtype=torch.bfloat16
    )
    x = torch.randn(2, 12, 128, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        ref_out = model(x).clone()

    cfg = FakeQuantConfig.qwen3_5_mxfp4_hybrid()
    model = prepare_qad_model(
        model,
        cfg,
        lora_config={"r": 8, "lora_alpha": 16, "target_modules": ["q_proj", "o_proj"], "bias": "none"},
        lora_merge_before_quant=True,
        lora_moe_experts=True,
    )
    experts = None
    for m in model.modules():
        if isinstance(m, QuantFusedMoEExperts):
            experts = m
        elif hasattr(m, "weight_quantizer"):
            for attr in ("weight_quantizer", "input_quantizer"):
                q = getattr(m, attr, None)
                if q is not None and q.is_enabled and not q.is_dynamic:
                    q.set_amax(torch.tensor(1.0, device=device))

    assert experts is not None and experts.has_lora
    assert experts.lora_gate_up_A.requires_grad and experts.lora_down_B.requires_grad
    assert not experts.gate_up_proj.requires_grad

    with torch.no_grad():
        experts.lora_gate_up_B.normal_(0.0, 0.05)
        experts.lora_down_B.normal_(0.0, 0.05)

    with _self_teacher_context(model):
        teacher_y = model(x)
    assert torch.allclose(teacher_y, ref_out, atol=2e-2, rtol=2e-2)

    student_y = model(x)
    student_y.sum().backward()
    assert experts.lora_gate_up_A.grad is not None and torch.isfinite(experts.lora_gate_up_A.grad).all()
    assert experts.lora_down_B.grad is not None and torch.isfinite(experts.lora_down_B.grad).all()

    model.eval()
    with torch.no_grad():
        before_merge = model(x)
        merge_lora_into_quant_linear(model)
        after_merge = model(x)
    assert not experts.has_lora
    assert torch.allclose(before_merge, after_merge, atol=1e-2, rtol=1e-2)


def test_moe_quantizer_state_and_compressed_tensors_export(tmp_path):
    torch.manual_seed(3)
    # 1. Static FP8 MoE quantizer state round-trip
    fp8_model = TinyMoEDecoderModel(hidden_dim=64, intermediate_dim=32, num_experts=3, top_k=2)
    fp8_cfg = FakeQuantConfig.fp8_static(backend="pytorch")
    prepare_qat(fp8_model, fp8_cfg)
    for m in fp8_model.modules():
        for slot in (
            "weight_quantizer",
            "input_quantizer",
            "gate_up_weight_quantizer",
            "gate_up_input_quantizer",
            "down_weight_quantizer",
            "down_input_quantizer",
        ):
            q = getattr(m, slot, None)
            if q is not None and q.is_enabled and not q.is_dynamic:
                q.set_amax(torch.tensor(2.5))

    qstate = get_quantizer_state_dict(fp8_model)
    assert any("mlp.experts.gate_up_weight_quantizer" in k for k in qstate)
    assert any("mlp.experts.down_input_quantizer" in k for k in qstate)

    fresh = TinyMoEDecoderModel(hidden_dim=64, intermediate_dim=32, num_experts=3, top_k=2)
    prepare_qat(fresh, fp8_cfg)
    report = load_quantizer_state_dict(fresh, qstate)
    assert len(report["missing"]) == 0

    # 2. Hybrid MXFP4+FP8 MoE 3D LoRA merge + compressed-tensors per-expert packing
    model = TinyMoEDecoderModel(hidden_dim=64, intermediate_dim=32, num_experts=3, top_k=2)
    cfg = FakeQuantConfig.qwen3_5_mxfp4_hybrid(backend="pytorch")
    prepare_qat(model, cfg)
    attach_moe_expert_lora(model, r=4, lora_alpha=8)
    for m in model.modules():
        for slot in ("weight_quantizer", "input_quantizer"):
            q = getattr(m, slot, None)
            if q is not None and q.is_enabled and not q.is_dynamic:
                q.set_amax(torch.tensor(2.5))

    merge_lora_into_quant_linear(model)
    tensors, quant_cfg = pack_model_to_compressed_tensors(model)
    assert quant_cfg["quant_method"] == "compressed-tensors"
    for e in range(3):
        assert f"mlp.experts.{e}.gate_proj.weight_packed" in tensors
        assert f"mlp.experts.{e}.gate_proj.weight_scale" in tensors
        assert f"mlp.experts.{e}.up_proj.weight_packed" in tensors
        assert f"mlp.experts.{e}.down_proj.weight_packed" in tensors
        w_deq = dequantize_compressed_tensors_layer(
            f"mlp.experts.{e}.gate_proj",
            tensors,
            "mxfp4-pack-quantized",
            original_shape=(32, 64),
        )
        assert w_deq.shape == (32, 64) and torch.isfinite(w_deq).all()
    assert "mlp.experts.gate_up_proj" not in tensors
    assert "mlp.experts.down_proj" not in tensors


def test_from_float_releases_original_experts_parameters_for_fsdp2():
    import weakref

    class CustomGatedExperts(DummyQwen3MoeExperts):
        def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
            gate, up = gate_up.chunk(2, dim=-1)
            return torch.nn.functional.silu(gate) * up

    orig = CustomGatedExperts(num_experts=4, hidden_dim=32, intermediate_dim=16)
    ref = weakref.ref(orig)
    qmod = QuantFusedMoEExperts.from_float(orig)
    assert len(orig._parameters) == 0
    assert len(orig._buffers) == 0
    assert getattr(qmod._custom_apply_gate, "__self__", None) is None
    del orig
    import gc

    gc.collect()
    assert ref() is None
    x = torch.randn(6, 32)
    top_idx = torch.randint(0, 4, (6, 2))
    top_w = torch.softmax(torch.randn(6, 2), dim=-1)
    y = qmod(x, top_idx, top_w)
    assert y.shape == (6, 32) and torch.isfinite(y).all()
