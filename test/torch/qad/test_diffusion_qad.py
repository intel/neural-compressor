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
"""Unit tests for Diffusion Transformer (DiT) Quantization-Aware Distillation (`DiffusionQADTrainer`)."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from neural_compressor.torch.algorithms.fake_quant import FakeQuantConfig, QuantLinear, QuantLoraLinear
from neural_compressor.torch.qad import (
    DiffusionQADConfig,
    DiffusionQADTrainer,
    DiffusionQADTrainingArguments,
    prepare_diffusion_qad_model,
)
from neural_compressor.torch.quantization import prepare_qat


class _DummyAttention(nn.Module):
    def __init__(self, dim: int, cond_dim: int) -> None:
        super().__init__()
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(cond_dim, dim, bias=False)
        self.to_v = nn.Linear(cond_dim, dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=False), nn.Dropout(0.0)])

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        q = self.to_q(x)
        k = self.to_k(cond)
        v = self.to_v(cond)
        scores = torch.matmul(q, k.transpose(-1, -2)) * (x.size(-1) ** -0.5)
        attn = torch.softmax(scores, dim=-1)
        return self.to_out[0](torch.matmul(attn, v))


class _DummyFFN(nn.Module):
    def __init__(self, dim: int, ffn_dim: int) -> None:
        super().__init__()
        proj_in = nn.Module()
        proj_in.proj = nn.Linear(dim, ffn_dim)
        self.net = nn.ModuleList([proj_in, nn.GELU(), nn.Linear(ffn_dim, dim)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net[2](self.net[1](self.net[0].proj(x)))


class _DummyDiTBlock(nn.Module):
    def __init__(self, dim: int, text_dim: int, ffn_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = _DummyAttention(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = _DummyAttention(dim, text_dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = _DummyFFN(dim, ffn_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        x = x + self.attn1(self.norm1(x) + temb, self.norm1(x))
        x = x + self.attn2(self.norm2(x), cond)
        x = x + self.ffn(self.norm3(x))
        return x


class DummyWanDiT(nn.Module):
    """Lightweight DiT matching `WanTransformer3DModel` module hierarchy and forward signature."""

    def __init__(
        self,
        in_channels: int = 8,
        dim: int = 64,
        text_dim: int = 32,
        ffn_dim: int = 128,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.patch_embedding = nn.Conv3d(in_channels, dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.condition_embedder = nn.Linear(1, dim)
        self.blocks = nn.ModuleList([_DummyDiTBlock(dim, text_dim, ffn_dim) for _ in range(num_layers)])
        self.proj_out = nn.Linear(dim, in_channels * 4)
        self.in_channels = in_channels
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        return_dict: bool = True,
    ):
        b, c, f, h, w = hidden_states.shape
        x = self.patch_embedding(hidden_states).flatten(2).transpose(1, 2).contiguous()
        t_in = (timestep.float() / 1000.0).view(b, 1, 1).to(x.dtype)
        temb = self.condition_embedder(t_in)
        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, encoder_hidden_states, temb, use_reentrant=False)
            else:
                x = blk(x, encoder_hidden_states, temb)
        out = self.proj_out(x)
        out = out.reshape(b, f, h // 2, w // 2, c, 1, 2, 2).permute(0, 4, 1, 5, 2, 6, 3, 7)
        out = out.reshape(b, c, f, h, w)
        if not return_dict:
            return (out,)
        return {"sample": out}


class _LatentDataset(Dataset):
    def __init__(self, n: int = 12, c: int = 8, f: int = 2, h: int = 4, w: int = 4, text_dim: int = 32) -> None:
        gen = torch.Generator().manual_seed(0)
        self.latents = torch.randn(n, c, f, h, w, generator=gen)
        self.cond = torch.randn(n, 6, text_dim, generator=gen)

    def __len__(self) -> int:
        return self.latents.size(0)

    def __getitem__(self, idx: int):
        return {"latents": self.latents[idx], "encoder_hidden_states": self.cond[idx]}


def test_diffusion_hybrid_preset_selective_quantization():
    dit = DummyWanDiT()
    cfg = FakeQuantConfig.diffusion_mxfp4_fp8_hybrid(backend="pytorch")
    prepare_qat(dit, cfg)

    # Attention projections -> static FP8
    assert isinstance(dit.blocks[0].attn1.to_q, QuantLinear)
    assert dit.blocks[0].attn1.to_q.weight_quantizer.format.name == "fp8_e4m3"
    assert not dit.blocks[0].attn1.to_q.input_quantizer.is_dynamic
    assert isinstance(dit.blocks[0].attn2.to_out[0], QuantLinear)

    # FFN projections -> dynamic MXFP4
    assert isinstance(dit.blocks[0].ffn.net[0].proj, QuantLinear)
    assert dit.blocks[0].ffn.net[0].proj.weight_quantizer.format.name == "mxfp4"
    assert dit.blocks[0].ffn.net[0].proj.input_quantizer.is_dynamic
    assert isinstance(dit.blocks[0].ffn.net[2], QuantLinear)

    # Stem, time modulator, and output head remain unquantized
    assert type(dit.patch_embedding) is nn.Conv3d
    assert type(dit.condition_embedder) is nn.Linear
    assert type(dit.proj_out) is nn.Linear


def test_diffusion_qad_trainer_self_teacher_lora_with_calib_and_feature_kd(tmp_path):
    torch.manual_seed(11)
    student = DummyWanDiT()
    base_snapshot = student.blocks[0].attn1.to_q.weight.detach().clone()

    qcfg = FakeQuantConfig.diffusion_mxfp4_fp8_hybrid(backend="pytorch")
    student = prepare_diffusion_qad_model(
        student,
        qcfg,
        lora_config={
            "r": 4,
            "lora_alpha": 8,
            "target_modules": ["to_q", "to_k", "to_v", "to_out.0", "ffn.net.0.proj", "ffn.net.2"],
            "bias": "none",
        },
        lora_merge_before_quant=True,
    )
    qlora_mods = [m for m in student.modules() if isinstance(m, QuantLoraLinear)]
    assert len(qlora_mods) == 3 * 10  # 3 blocks * (4 self-attn + 4 cross-attn + 2 ffn)

    args = DiffusionQADTrainingArguments(
        output_dir=str(tmp_path / "diff_qad"),
        max_steps=3,
        per_device_train_batch_size=2,
        learning_rate=5e-3,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        use_cpu=True,
        gradient_checkpointing=True,
        calib_batches=2,
        remove_unused_columns=False,
    )
    dcfg = DiffusionQADConfig(
        teacher="self",
        teacher_device="self",
        loss_type="mse",
        kd_weight=1.0,
        task_weight=0.1,
        feature_kd_weight=0.1,
        feature_block_indices=[0, 2],
        prediction_type="flow_matching",
        flow_shift=5.0,
    )

    trainer = DiffusionQADTrainer(
        model=student,
        args=args,
        teacher_model="self",
        diffusion_config=dcfg,
        quant_config=qcfg,
        train_dataset=_LatentDataset(),
    )
    result = trainer.train()
    assert result.global_step == 3
    assert torch.isfinite(torch.tensor(result.training_loss))

    # Static FP8 quantizers must all be calibrated automatically before step 1
    for m in qlora_mods:
        for q in (m.weight_quantizer, m.input_quantizer):
            if q is not None and q.is_enabled and not q.is_dynamic:
                assert q.amax is not None and q.amax.numel() > 0

    # LoRA B weights updated while frozen base weights remain untouched!
    assert any(m.lora_B["default"].weight.abs().max().item() > 0 for m in qlora_mods)
    after_base = qlora_mods[0].base_layer.weight.detach()
    assert torch.equal(base_snapshot, after_base)


def test_diffusion_qad_trainer_external_teacher_cosine_mse(tmp_path):
    torch.manual_seed(21)
    teacher = DummyWanDiT()
    student = copy.deepcopy(teacher)
    qcfg = FakeQuantConfig.diffusion_mxfp4_fp8_hybrid(backend="pytorch")
    prepare_qat(student, qcfg)

    args = DiffusionQADTrainingArguments(
        output_dir=str(tmp_path / "diff_ext"),
        max_steps=2,
        per_device_train_batch_size=2,
        learning_rate=1e-3,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        use_cpu=True,
        calib_batches=1,
        remove_unused_columns=False,
    )
    dcfg = DiffusionQADConfig(
        teacher=teacher,
        teacher_device="cpu",
        loss_type="cosine_mse",
        kd_weight=1.0,
        prediction_type="v_prediction",
    )
    trainer = DiffusionQADTrainer(
        model=student,
        args=args,
        teacher_model=teacher,
        diffusion_config=dcfg,
        quant_config=qcfg,
        train_dataset=_LatentDataset(),
    )
    res = trainer.train()
    assert res.global_step == 2
    assert torch.isfinite(torch.tensor(res.training_loss))
