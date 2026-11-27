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
"""Fused 3D Mixture-of-Experts (MoE) fake-quantization plugin (`QuantFusedMoEExperts`).

Supports HuggingFace ``transformers >= 5.0`` fused 3D ``*Experts`` modules
(e.g. ``Qwen3MoeExperts``, ``Qwen3_5MoeExperts``, ``Qwen2MoeExperts``, ``Llama4TextExperts``,
``MixtralExperts``) that store routed expert projections as 3D ``nn.Parameter`` tensors
(``gate_up_proj: [E, 2*I, H]`` and ``down_proj: [E, H, I]``) rather than ``nn.ModuleList[nn.Linear]``.

Key training invariants:
1. Zero Python per-expert loops on CUDA BF16 via ``torch.nn.functional.grouped_mm`` + ``histc`` offsets.
2. Batched Triton QDQ: only 4 QDQ launches per MoE block (gate_up input/weight + down input/weight)
   instead of ``4 * num_experts`` launches.
3. Optional 3D Grouped Path-B LoRA (``enable_lora`` / ``attach_moe_expert_lora``) with ``dim 0 = E``
   for native PyTorch FSDP2 sharding and zero-extra-VRAM ``teacher="self"`` toggling.
"""

from __future__ import annotations

import importlib
import math
from typing import Callable, Literal, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModuleQuantConfig
from ..conversion import register_quant_module
from ..formats import E2M1_MAX, E4M3_MAX
from ..functional import _e2m1_qdq_scaled
from ..tensor_quantizer import TensorQuantizer

QuantOrder = Literal["lora_then_quant", "quant_then_lora"]

_MOE_REGISTERED_CLASSES: set[Type[nn.Module]] = set()
_KNOWN_HF_EXPERTS_SPECS: tuple[tuple[str, str], ...] = (
    ("transformers.models.qwen3_moe.modeling_qwen3_moe", "Qwen3MoeExperts"),
    ("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe", "Qwen3_5MoeExperts"),
    ("transformers.models.qwen2_moe.modeling_qwen2_moe", "Qwen2MoeExperts"),
    ("transformers.models.mixtral.modeling_mixtral", "MixtralExperts"),
    ("transformers.models.llama4.modeling_llama4", "Llama4TextExperts"),
    ("transformers.models.glm4_moe.modeling_glm4_moe", "Glm4MoeExperts"),
    ("transformers.models.ernie4_5_moe.modeling_ernie4_5_moe", "Ernie4_5_MoeExperts"),
    ("transformers.models.dots1.modeling_dots1", "Dots1Experts"),
    ("transformers.models.minimax_m2.modeling_minimax_m2", "MiniMaxM2Experts"),
)


def is_fused_moe_experts_module(mod: nn.Module) -> bool:
    """Return True if ``mod`` is an unwrapped 3D fused MoE experts module."""
    if getattr(mod, "_is_fake_quant_module", False):
        return False
    down = getattr(mod, "down_proj", None)
    up_or_gate = getattr(mod, "gate_up_proj", getattr(mod, "up_proj", None))
    return (
        isinstance(down, torch.Tensor)
        and isinstance(up_or_gate, torch.Tensor)
        and down.ndim == 3
        and up_or_gate.ndim == 3
        and hasattr(mod, "num_experts")
        and hasattr(mod, "act_fn")
    )


def is_fused_moe_experts_class(cls: Type[nn.Module]) -> bool:
    """Return True if ``cls`` is a known or registered fused 3D MoE experts class."""
    if getattr(cls, "_is_fake_quant_module", False):
        return False
    if cls in _MOE_REGISTERED_CLASSES:
        return True
    mod_name = getattr(cls, "__module__", "")
    cls_name = getattr(cls, "__name__", "")
    return mod_name.startswith("transformers.") and cls_name.endswith("Experts")


def _ensure_contiguous_grad(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.requires_grad:
        tensor.register_hook(lambda g: g if g.is_contiguous() else g.contiguous())
    return tensor


def _grouped_linear(
    input_2d: torch.Tensor,
    weight_3d: torch.Tensor,
    offs: torch.Tensor,
    *,
    bias: Optional[torch.Tensor] = None,
    is_transposed: bool = False,
) -> torch.Tensor:
    """Compute grouped per-expert GEMM ``input_2d @ W_e^T`` (or ``@ W_e`` when ``is_transposed``)."""
    use_grouped_mm = (
        hasattr(F, "grouped_mm")
        and input_2d.is_cuda
        and input_2d.dtype == torch.bfloat16
        and weight_3d.dtype == torch.bfloat16
    )
    if use_grouped_mm:
        inp_c = input_2d if input_2d.is_contiguous() else input_2d.contiguous()
        rhs = weight_3d if is_transposed else weight_3d.transpose(-2, -1)
        out = F.grouped_mm(inp_c, rhs, offs=offs)
        out = _ensure_contiguous_grad(out)
    else:
        out_dim = weight_3d.size(-1) if is_transposed else weight_3d.size(-2)
        out = input_2d.new_empty((input_2d.size(0), out_dim))
        offs_cpu = offs.detach().to("cpu", dtype=torch.int64).tolist()
        start = 0
        for e_idx, end in enumerate(offs_cpu):
            if end > start:
                w_e = weight_3d[e_idx]
                inp_e = input_2d[start:end]
                if w_e.dtype != inp_e.dtype:
                    w_e = w_e.to(inp_e.dtype)
                out[start:end] = inp_e @ w_e if is_transposed else F.linear(inp_e, w_e)
            start = end

    if bias is not None:
        out = out + bias.to(out.dtype)
    return out


def _per_expert_dynamic_qdq_no_grad(w: torch.Tensor, fmt_name: str) -> torch.Tensor:
    """Per-expert dynamic global-scale QDQ for ``nvfp4`` or dynamic ``fp8_e4m3`` on ``[E, Out, In]``."""
    wf = w.float()
    if fmt_name == "fp8_e4m3":
        amax = wf.abs().amax(dim=(-2, -1), keepdim=True)
        raw_scale = amax / E4M3_MAX
        safe_scale = torch.where(raw_scale > 0, raw_scale, torch.ones_like(raw_scale))
        q = (wf / safe_scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        return (q.float() * safe_scale).to(w.dtype)
    if fmt_name == "nvfp4":
        e_dim, out_dim, in_dim = wf.shape
        if in_dim % 16 != 0:
            raise ValueError(f"NVFP4 requires last dim divisible by 16, got {in_dim}")
        global_amax = wf.abs().amax(dim=(-2, -1), keepdim=True)
        raw_global = global_amax / (E2M1_MAX * E4M3_MAX)
        global_scale = torch.where(raw_global > 0, raw_global, torch.ones_like(raw_global)).unsqueeze(-1)
        blocks = wf.reshape(e_dim, out_dim, in_dim // 16, 16)
        block_amax = blocks.abs().amax(dim=-1, keepdim=True)
        block_scale = (block_amax / (E2M1_MAX * global_scale)).clamp(2.0**-9, E4M3_MAX)
        block_scale = block_scale.to(torch.float8_e4m3fn).float()
        eff_scale = block_scale * global_scale
        q = _e2m1_qdq_scaled(blocks / eff_scale) * eff_scale
        return q.reshape(w.shape).to(w.dtype)
    raise ValueError(f"Unexpected per-expert dynamic format {fmt_name!r}")


def _quantize_3d_expert_weight(
    quantizer: TensorQuantizer,
    weight: torch.Tensor,
    *,
    is_transposed: bool = False,
) -> torch.Tensor:
    """Apply ``quantizer`` along the reduction channel of a 3D ``[E, Out, In]`` (or transposed) weight."""
    if not quantizer.is_enabled and not quantizer.is_calibrating:
        return weight
    w = weight.transpose(-2, -1).contiguous() if is_transposed else weight
    if quantizer.is_calibrating:
        quantizer.observe(w)
        if not quantizer.is_enabled:
            return weight
    if not quantizer.is_dynamic or quantizer.format.name in ("mxfp4", "mxfp8", "bf16"):
        qw = quantizer(w)
    else:
        with torch.no_grad():
            q_det = _per_expert_dynamic_qdq_no_grad(w.detach(), quantizer.format.name)
        qw = w + (q_det - w).detach() if (w.requires_grad and torch.is_grad_enabled()) else q_det
    return qw.transpose(-2, -1).contiguous() if is_transposed else qw


class QuantFusedMoEExperts(nn.Module):
    """Fake-quantized fused 3D MoE experts replacement for ``transformers`` ``*Experts`` modules."""

    _is_fake_quant_module: bool = True

    def __init__(
        self,
        *,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor],
        config: ModuleQuantConfig,
        has_gate: bool = True,
        is_transposed: bool = False,
        has_bias: bool = False,
    ) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.hidden_dim = int(hidden_dim)
        self.intermediate_dim = int(intermediate_dim)
        self.has_gate = bool(has_gate)
        self.is_transposed = bool(is_transposed)
        self.has_bias = bool(has_bias)
        self.act_fn = act_fn
        self.quant_config = config
        self._custom_apply_gate: Optional[Callable[..., torch.Tensor]] = None
        self._custom_apply_gate_unbound: bool = False
        self.swiglu_alpha: Optional[float] = None
        self.swiglu_limit: Optional[float] = None
        self.limit: Optional[float] = None

        self.gate_up_weight_quantizer = TensorQuantizer(config.weights)
        self.gate_up_input_quantizer = TensorQuantizer(config.input_activations)
        self.down_weight_quantizer = TensorQuantizer(config.weights)
        self.down_input_quantizer = TensorQuantizer(config.input_activations)
        self.output_quantizer = TensorQuantizer(config.output_activations)

        self._disable_adapters: bool = False
        self.lora_rank: int = 0
        self.lora_scaling: float = 0.0
        self.quant_order: QuantOrder = "lora_then_quant"
        self._source_cls: Optional[Type[nn.Module]] = None

    @classmethod
    def from_float(cls, mod: nn.Module, config: Optional[ModuleQuantConfig] = None) -> "QuantFusedMoEExperts":
        """Wrap a 3D fused ``*Experts`` module sharing its parameters in-place."""
        if not is_fused_moe_experts_module(mod):
            raise TypeError(
                f"QuantFusedMoEExperts.from_float expects a 3D fused Experts module, got {type(mod).__name__}"
            )
        resolved = config if config is not None else ModuleQuantConfig()
        has_gate = hasattr(mod, "gate_up_proj") and isinstance(mod.gate_up_proj, torch.Tensor)
        up_param: torch.Tensor = mod.gate_up_proj if has_gate else mod.up_proj
        down_param: torch.Tensor = mod.down_proj
        is_transposed = bool(getattr(mod, "is_transposed", False))
        has_bias = bool(getattr(mod, "has_bias", False))

        num_experts = int(getattr(mod, "num_experts", down_param.size(0)))
        if is_transposed:
            hidden_dim = int(getattr(mod, "hidden_dim", getattr(mod, "hidden_size", down_param.size(-1))))
            intermediate_dim = int(
                getattr(mod, "intermediate_dim", getattr(mod, "moe_intermediate_size", down_param.size(-2)))
            )
        else:
            hidden_dim = int(getattr(mod, "hidden_dim", getattr(mod, "hidden_size", down_param.size(-2))))
            intermediate_dim = int(
                getattr(mod, "intermediate_dim", getattr(mod, "moe_intermediate_size", down_param.size(-1)))
            )

        qmod = cls(
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            act_fn=mod.act_fn,
            config=resolved,
            has_gate=has_gate,
            is_transposed=is_transposed,
            has_bias=has_bias,
        )
        qmod._source_cls = type(mod)

        if has_gate:
            qmod.gate_up_proj = up_param
            if has_bias and hasattr(mod, "gate_up_proj_bias"):
                qmod.gate_up_proj_bias = mod.gate_up_proj_bias
        else:
            qmod.up_proj = up_param
            if has_bias and hasattr(mod, "up_proj_bias"):
                qmod.up_proj_bias = mod.up_proj_bias
        qmod.down_proj = down_param
        if has_bias and hasattr(mod, "down_proj_bias"):
            qmod.down_proj_bias = mod.down_proj_bias

        for attr in ("swiglu_alpha", "swiglu_limit", "limit"):
            if hasattr(mod, attr):
                setattr(qmod, attr, getattr(mod, attr))

        if "_apply_gate" in mod.__dict__ and callable(mod.__dict__["_apply_gate"]):
            qmod._custom_apply_gate = mod.__dict__["_apply_gate"]
            qmod._custom_apply_gate_unbound = False
        else:
            cls_gate = getattr(type(mod), "_apply_gate", None)
            if callable(cls_gate):
                qmod._custom_apply_gate = cls_gate
                qmod._custom_apply_gate_unbound = True
            else:
                qmod._custom_apply_gate_unbound = False

        # Release parameter/buffer dicts on the replaced source module so FSDP2's in-place
        # `param.data = param.to(device)` during sharding never pins unsharded tensors via `mod`.
        mod._parameters.clear()
        mod._buffers.clear()

        param_device = down_param.device
        if param_device.type != "meta":
            for q in (
                qmod.gate_up_weight_quantizer,
                qmod.gate_up_input_quantizer,
                qmod.down_weight_quantizer,
                qmod.down_input_quantizer,
                qmod.output_quantizer,
            ):
                if q.amax is not None:
                    q.amax = q.amax.to(device=param_device, dtype=torch.float32)
        return qmod

    # ------------------------------------------------------------------
    # Optional 3D Grouped Path-B / Path-A LoRA
    # ------------------------------------------------------------------

    @property
    def has_lora(self) -> bool:
        return self.lora_rank > 0 and hasattr(self, "lora_down_A")

    def enable_lora(
        self,
        r: int,
        lora_alpha: float = 16.0,
        *,
        use_rslora: bool = False,
        quant_order: QuantOrder = "lora_then_quant",
    ) -> None:
        """Attach 3D grouped LoRA parameters shardable along ``dim=0 (num_experts)`` under FSDP2."""
        if r <= 0:
            raise ValueError(f"LoRA rank r must be positive, got {r}")
        if quant_order not in ("lora_then_quant", "quant_then_lora"):
            raise ValueError(f"Unknown quant_order {quant_order!r}")

        up_base = self.gate_up_proj if self.has_gate else self.up_proj
        up_base.requires_grad_(False)
        self.down_proj.requires_grad_(False)

        dtype = up_base.dtype
        device = up_base.device
        up_out = (2 * self.intermediate_dim) if self.has_gate else self.intermediate_dim

        lora_gu_a = torch.empty((self.num_experts, r, self.hidden_dim), dtype=dtype, device=device)
        lora_gu_b = torch.zeros((self.num_experts, up_out, r), dtype=dtype, device=device)
        lora_dn_a = torch.empty((self.num_experts, r, self.intermediate_dim), dtype=dtype, device=device)
        lora_dn_b = torch.zeros((self.num_experts, self.hidden_dim, r), dtype=dtype, device=device)
        if device.type != "meta":
            nn.init.kaiming_uniform_(lora_gu_a, a=math.sqrt(5))
            nn.init.kaiming_uniform_(lora_dn_a, a=math.sqrt(5))

        self.lora_gate_up_A = nn.Parameter(lora_gu_a, requires_grad=True)
        self.lora_gate_up_B = nn.Parameter(lora_gu_b, requires_grad=True)
        self.lora_down_A = nn.Parameter(lora_dn_a, requires_grad=True)
        self.lora_down_B = nn.Parameter(lora_dn_b, requires_grad=True)
        self.lora_rank = int(r)
        self.lora_scaling = float(lora_alpha) / (math.sqrt(r) if use_rslora else float(r))
        self.quant_order = quant_order

    def _lora_delta(self, lora_a: torch.Tensor, lora_b: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        delta = torch.bmm(self.lora_scaling * lora_b, lora_a)
        if self.is_transposed:
            delta = delta.transpose(-2, -1)
        return delta.to(device=target.device, dtype=target.dtype)

    def effective_gate_up_weight(self) -> torch.Tensor:
        base = self.gate_up_proj if self.has_gate else self.up_proj
        if not self.has_lora or self._disable_adapters:
            return base
        return base + self._lora_delta(self.lora_gate_up_A, self.lora_gate_up_B, base)

    def effective_down_weight(self) -> torch.Tensor:
        base = self.down_proj
        if not self.has_lora or self._disable_adapters:
            return base
        return base + self._lora_delta(self.lora_down_A, self.lora_down_B, base)

    @torch.no_grad()
    def merge_lora_(self) -> None:
        """Fold active 3D LoRA adapters into base 3D expert weights in-place and remove LoRA params."""
        if not self.has_lora:
            return
        saved_disable = self._disable_adapters
        self._disable_adapters = False
        try:
            if self.has_gate:
                self.gate_up_proj.copy_(self.effective_gate_up_weight())
            else:
                self.up_proj.copy_(self.effective_gate_up_weight())
            self.down_proj.copy_(self.effective_down_weight())
        finally:
            self._disable_adapters = saved_disable

        for attr in ("lora_gate_up_A", "lora_gate_up_B", "lora_down_A", "lora_down_B"):
            if hasattr(self, attr):
                delattr(self, attr)
        self.lora_rank = 0
        self.lora_scaling = 0.0

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _apply_gate(self, proj_out: torch.Tensor) -> torch.Tensor:
        if self._custom_apply_gate is not None:
            if self._custom_apply_gate_unbound:
                return self._custom_apply_gate(self, proj_out)
            return self._custom_apply_gate(proj_out)
        gate, up = proj_out.chunk(2, dim=-1)
        return self.act_fn(gate) * up

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_dim = orig_shape[-1]
        hidden_2d = hidden_states.reshape(-1, hidden_dim)
        num_tokens = hidden_2d.size(0)
        num_top_k = top_k_index.size(-1)
        device = hidden_2d.device

        if num_tokens == 0 or num_top_k == 0:
            return self.output_quantizer(hidden_2d.clone()).reshape(orig_shape)

        sample_weights = top_k_weights.reshape(-1)
        expert_ids = top_k_index.reshape(-1)

        expert_ids_g, perm = torch.sort(expert_ids)
        selected_hidden_g = hidden_2d[perm // num_top_k]
        sample_weights_g = sample_weights[perm]

        histc_in = expert_ids_g.float() if device.type in ("cpu", "mps") else expert_ids_g.int()
        tokens_per_expert = torch.histc(histc_in, bins=self.num_experts, min=0, max=self.num_experts - 1)
        offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)

        sentinel_mask = (expert_ids_g >= self.num_experts).unsqueeze(-1)
        expert_ids_clamped = expert_ids_g.clamp(min=0, max=self.num_experts - 1)
        selected_hidden_g = selected_hidden_g.masked_fill(sentinel_mask, 0.0)

        use_path_a = self.has_lora and not self._disable_adapters and self.quant_order == "quant_then_lora"
        raw_gu_w = (
            (self.gate_up_proj if self.has_gate else self.up_proj) if use_path_a else self.effective_gate_up_weight()
        )
        q_gu_in = self.gate_up_input_quantizer(selected_hidden_g)
        q_gu_w = _quantize_3d_expert_weight(self.gate_up_weight_quantizer, raw_gu_w, is_transposed=self.is_transposed)
        if q_gu_w.dtype != q_gu_in.dtype:
            q_gu_w = q_gu_w.to(q_gu_in.dtype)

        gu_bias: Optional[torch.Tensor] = None
        if self.has_bias:
            bias_src = getattr(self, "gate_up_proj_bias" if self.has_gate else "up_proj_bias", None)
            if bias_src is not None:
                gu_bias = bias_src[expert_ids_clamped]

        proj_out = _grouped_linear(q_gu_in, q_gu_w, offsets, bias=gu_bias, is_transposed=self.is_transposed)
        if use_path_a:
            delta_gu = self._lora_delta(self.lora_gate_up_A, self.lora_gate_up_B, selected_hidden_g)
            proj_out = proj_out + _grouped_linear(
                selected_hidden_g, delta_gu, offsets, is_transposed=self.is_transposed
            )

        interm = self._apply_gate(proj_out) if self.has_gate else self.act_fn(proj_out)

        raw_dn_w = self.down_proj if use_path_a else self.effective_down_weight()
        q_dn_in = self.down_input_quantizer(interm)
        q_dn_w = _quantize_3d_expert_weight(self.down_weight_quantizer, raw_dn_w, is_transposed=self.is_transposed)
        if q_dn_w.dtype != q_dn_in.dtype:
            q_dn_w = q_dn_w.to(q_dn_in.dtype)

        dn_bias: Optional[torch.Tensor] = None
        if self.has_bias:
            dn_bias_src = getattr(self, "down_proj_bias", None)
            if dn_bias_src is not None:
                dn_bias = dn_bias_src[expert_ids_clamped]

        down_out = _grouped_linear(q_dn_in, q_dn_w, offsets, bias=dn_bias, is_transposed=self.is_transposed)
        if use_path_a:
            delta_dn = self._lora_delta(self.lora_down_A, self.lora_down_B, interm)
            down_out = down_out + _grouped_linear(interm, delta_dn, offsets, is_transposed=self.is_transposed)

        weighted_out = down_out * sample_weights_g.unsqueeze(-1)
        weighted_out = weighted_out.masked_fill(sentinel_mask, 0.0)

        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(perm.size(0), device=device)
        weighted_out = weighted_out[inv_perm]

        final_hidden = weighted_out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1).to(hidden_states.dtype)
        return self.output_quantizer(final_hidden).reshape(orig_shape)

    def extra_repr(self) -> str:
        parts = [
            f"num_experts={self.num_experts}",
            f"hidden_dim={self.hidden_dim}",
            f"intermediate_dim={self.intermediate_dim}",
            f"weights={self.gate_up_weight_quantizer.format.name}",
            f"inputs={self.gate_up_input_quantizer.format.name}",
        ]
        if self.has_lora:
            parts.append(f"lora_rank={self.lora_rank}")
        return ", ".join(parts)


def register_moe_quant_modules() -> list[Type[nn.Module]]:
    """Register available ``transformers`` 3D ``*Experts`` classes in ``fake_quant``'s module registry."""
    newly_registered: list[Type[nn.Module]] = []
    for mod_path, cls_name in _KNOWN_HF_EXPERTS_SPECS:
        try:
            py_mod = importlib.import_module(mod_path)
            cls = getattr(py_mod, cls_name, None)
        except Exception:
            cls = None
        if isinstance(cls, type) and issubclass(cls, nn.Module) and cls not in _MOE_REGISTERED_CLASSES:
            register_quant_module(cls, QuantFusedMoEExperts.from_float)
            _MOE_REGISTERED_CLASSES.add(cls)
            newly_registered.append(cls)
    return newly_registered


def attach_moe_expert_lora(
    model: nn.Module,
    r: int = 16,
    lora_alpha: float = 32.0,
    *,
    use_rslora: bool = False,
    quant_order: QuantOrder = "lora_then_quant",
) -> int:
    """Attach 3D grouped LoRA adapters to all :class:`QuantFusedMoEExperts` modules in ``model``."""
    count = 0
    for mod in model.modules():
        if isinstance(mod, QuantFusedMoEExperts):
            mod.enable_lora(r=r, lora_alpha=lora_alpha, use_rslora=use_rslora, quant_order=quant_order)
            count += 1
    return count


__all__ = [
    "QuantFusedMoEExperts",
    "attach_moe_expert_lora",
    "is_fused_moe_experts_class",
    "is_fused_moe_experts_module",
    "register_moe_quant_modules",
]
