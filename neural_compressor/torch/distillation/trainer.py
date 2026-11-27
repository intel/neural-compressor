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
"""Standalone Knowledge Distillation Trainer with zero full-vocab logit materialization."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from typing import Any, Iterator, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer

from .config import DistillationConfig
from .losses import full_vocab_kl, topk_kl, topk_kl_from_topk


def _unwrap_model(model: Any) -> nn.Module:
    """Unwrap DDP / FSDP ``.module`` wrappers to access the inner ``nn.Module``."""
    while hasattr(model, "module") and isinstance(getattr(model, "module"), nn.Module):
        model = model.module
    return model


@contextmanager
def _disable_quantizers(model: nn.Module) -> Iterator[None]:
    """Temporarily disable any fake-quantizer modules (duck-typed without importing ``quantization``)."""
    toggled_methods: list[tuple[nn.Module, str]] = []
    for mod in model.modules():
        if not hasattr(mod, "is_enabled") or not bool(mod.is_enabled):
            continue
        if callable(getattr(mod, "disable_quant", None)) and callable(getattr(mod, "enable_quant", None)):
            toggled_methods.append((mod, "quant"))
            mod.disable_quant()
        elif callable(getattr(mod, "disable", None)) and callable(getattr(mod, "enable", None)):
            toggled_methods.append((mod, "plain"))
            mod.disable()
    try:
        yield
    finally:
        for mod, kind in toggled_methods:
            if kind == "quant":
                mod.enable_quant()
            else:
                mod.enable()


def _find_decoder_layers(model: nn.Module) -> Optional[nn.ModuleList]:
    """Locate a decoder ``nn.ModuleList`` on ``model`` (walking standard HF / VLM attributes)."""
    for list_attr in ("layers", "h", "blocks", "block"):
        layers = getattr(model, list_attr, None)
        if isinstance(layers, nn.ModuleList) and len(layers) > 0:
            return layers
    for attr in ("language_model", "base_model", "model", "transformer", "module"):
        sub = getattr(model, attr, None)
        if isinstance(sub, nn.Module) and sub is not model:
            found = _find_decoder_layers(sub)
            if found is not None:
                return found
    return None


def _place_teacher_model(teacher: nn.Module, placement: str, device: torch.device) -> nn.Module:
    """Place ``teacher`` on ``cpu``, local ``device``, or FSDP2 ``shard`` across initialized ranks."""
    teacher.eval()
    teacher.requires_grad_(False)
    already_sharded = any(hasattr(p, "full_tensor") or hasattr(p, "ds_id") for p in teacher.parameters())
    if already_sharded:
        return teacher

    if placement == "cpu":
        return teacher.to(torch.device("cpu"))

    world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    should_shard = placement in ("shard", "sharded") or (
        placement == "auto" and sum(p.numel() for p in teacher.parameters()) >= 8e9
    )
    if should_shard and world > 1 and device.type != "cpu":
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        sample_param = next(teacher.parameters(), None)
        param_dtype = (
            sample_param.dtype
            if sample_param is not None and sample_param.dtype in (torch.bfloat16, torch.float16)
            else torch.bfloat16
        )
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=torch.float32,
            output_dtype=param_dtype,
        )
        layers = _find_decoder_layers(teacher)
        if layers is not None:
            for layer in layers:
                fully_shard(layer, mp_policy=mp_policy, reshard_after_forward=True)
        fully_shard(teacher, mp_policy=mp_policy, reshard_after_forward=True)
        return teacher.to(device)

    return teacher.to(device)


def _core_pretrained_model(model: nn.Module) -> nn.Module:
    """Walk DDP/FSDP and PEFT wrappers to find the underlying ``PreTrainedModel``."""
    cur = _unwrap_model(model)
    if hasattr(cur, "peft_config") and hasattr(cur, "base_model") and hasattr(cur.base_model, "model"):
        cur = cur.base_model.model
    return cur


def _locate_lm_head(model: nn.Module) -> Optional[nn.Module]:
    """Locate the vocabulary output projection ``nn.Module`` via ``get_output_embeddings`` or ``.lm_head``."""
    core = _core_pretrained_model(model)
    if hasattr(core, "get_output_embeddings") and callable(core.get_output_embeddings):
        try:
            head = core.get_output_embeddings()
            if isinstance(head, nn.Module):
                return head
        except (AttributeError, NotImplementedError):
            pass
    head = getattr(core, "lm_head", None)
    return head if isinstance(head, nn.Module) else None


@contextmanager
def _identity_lm_head(lm_head: nn.Module) -> Iterator[None]:
    """Temporarily turn ``lm_head.forward`` into an identity map so ``PreTrainedModel.forward`` returns
    ``last_hidden_state`` ``[B, S, H]`` without allocating ``[B, S, V]`` logits or 64-layer hidden tuples."""
    orig_forward = lm_head.forward

    def _passthrough(hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return hidden_states

    lm_head.forward = _passthrough  # type: ignore[method-assign]
    try:
        yield
    finally:
        lm_head.forward = orig_forward  # type: ignore[method-assign]


def _needs_fsdp_or_zero_redirect(lm_head: nn.Module) -> bool:
    """Return True if ``lm_head.weight`` is sharded by FSDP2 ``DTensor`` or DeepSpeed ZeRO-3."""
    weight = getattr(lm_head, "weight", None)
    if weight is None:
        return False
    return hasattr(weight, "full_tensor") or hasattr(weight, "ds_id")


def _forward_redirect(
    wrapped_model: nn.Module,
    target_submodule: nn.Module,
    chunk: torch.Tensor,
) -> torch.Tensor:
    """Invoke ``target_submodule(chunk)`` through ``wrapped_model.forward`` so root FSDP2 / ZeRO-3
    pre-forward unshard and backward reduce-scatter hooks execute properly."""
    inner = _unwrap_model(wrapped_model)
    orig_forward = inner.forward

    def _redirected_forward(*args, **kwargs):
        x = args[0]
        w = getattr(target_submodule, "weight", None)
        if (
            isinstance(w, torch.Tensor)
            and w.is_floating_point()
            and not hasattr(w, "full_tensor")
            and x.dtype != w.dtype
        ):
            x = x.to(w.dtype)
        return target_submodule(x, *args[1:], **kwargs)

    inner.forward = _redirected_forward  # type: ignore[method-assign]
    try:
        return wrapped_model(chunk)
    finally:
        inner.forward = orig_forward  # type: ignore[method-assign]


@contextmanager
def _self_teacher_context(model: nn.Module) -> Iterator[None]:
    """Temporarily disable all quantizer modules and PEFT LoRA adapters for ``teacher='self'``.

    Note: Toggles ``mod._disable_adapters = True`` directly under ``torch.no_grad()`` rather than
    calling ``PeftModel.disable_adapter()``, because ``disable_adapter()`` mutates
    ``param.requires_grad = False`` which corrupts PyTorch FSDP2's lazy ``_init_mp_dtypes()`` on
    the first forward pass.
    """
    inner = _unwrap_model(model)
    was_training = inner.training
    toggled_lora_layers: list[tuple[nn.Module, bool]] = []
    with ExitStack() as stack:
        stack.enter_context(torch.no_grad())
        stack.enter_context(_disable_quantizers(inner))
        for mod in inner.modules():
            if hasattr(mod, "_disable_adapters"):
                toggled_lora_layers.append((mod, bool(mod._disable_adapters)))
                mod._disable_adapters = True
        if not toggled_lora_layers and hasattr(inner, "disable_adapter") and callable(inner.disable_adapter):
            stack.enter_context(inner.disable_adapter())
        inner.eval()
        try:
            yield
        finally:
            for mod, prev in toggled_lora_layers:
                mod._disable_adapters = prev
            if was_training:
                inner.train()


class KDTrainer(Trainer):
    """HuggingFace ``Trainer`` subclass implementing chunked Top-K KL / Full KL + Hybrid CE distillation.

    Supports:
    * Standard ``PreTrainedModel``, VLM ``ForConditionalGeneration``, ``PeftModel``, FSDP2, and DDP.
    * Zero full-sequence ``[B, S, V]`` logit allocation via ``_identity_lm_head`` + chunked ``lm_head`` projection.
    * Three teacher modes: online external ``teacher_model``, zero-VRAM LoRA ``teacher_model="self"``, and
      offline precomputed ``teacher_topk_{values,indices}`` batches.
    """

    def __init__(
        self,
        model: Union[nn.Module, Any] = None,
        args: Any = None,
        *pos_args,
        teacher_model: Optional[Union[nn.Module, str]] = None,
        teacher: Optional[Union[nn.Module, str]] = None,
        distill_config: Optional[DistillationConfig] = None,
        **kwargs,
    ) -> None:
        resolved_teacher = teacher_model if teacher_model is not None else teacher
        if distill_config is None:
            distill_config = DistillationConfig(teacher=resolved_teacher)
        if resolved_teacher is None and distill_config.teacher is not None:
            if isinstance(distill_config.teacher, nn.Module) or distill_config.teacher == "self":
                resolved_teacher = distill_config.teacher

        self.distill_config: DistillationConfig = distill_config
        self.distill: DistillationConfig = distill_config
        self.self_teacher: bool = resolved_teacher == "self" or distill_config.teacher_device == "self"

        if self.self_teacher:
            self.teacher: Optional[nn.Module] = None
        elif isinstance(resolved_teacher, nn.Module):
            self.teacher = resolved_teacher
            self.teacher.eval()
            self.teacher.requires_grad_(False)
        else:
            self.teacher = None

        self._last_loss_parts: dict[str, float] = {}
        super().__init__(model=model, args=args, *pos_args, **kwargs)
        if self.teacher is not None:
            self.teacher = _place_teacher_model(self.teacher, self.distill.teacher_device, self.args.device)

    def _extract_last_hidden_state(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the transformer stack through ``model.forward`` and return ``[B, S, H]`` without ``[B, S, V]`` logits."""
        core = _core_pretrained_model(model)
        fwd_kwargs: dict[str, Any] = {"input_ids": input_ids}
        if attention_mask is not None:
            fwd_kwargs["attention_mask"] = attention_mask
        if position_ids is not None:
            fwd_kwargs["position_ids"] = position_ids

        # Duck-typed fallback if a custom module provides explicit forward_hidden / logits hooks.
        if hasattr(core, "forward_hidden") and hasattr(core, "logits"):
            return model(**fwd_kwargs)

        lm_head = _locate_lm_head(model)
        fwd_kwargs["use_cache"] = False
        if lm_head is not None:
            with _identity_lm_head(lm_head):
                outputs = model(**fwd_kwargs)
            hidden = (
                outputs.logits
                if hasattr(outputs, "logits")
                else (outputs[0] if isinstance(outputs, tuple) else outputs)
            )
            return hidden

        outputs = model(**fwd_kwargs, output_hidden_states=True)
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise TypeError(
                f"{type(core).__name__} neither exposes an output embedding (`lm_head`) nor returns `hidden_states`."
            )
        return hidden_states[-1]

    # Backward-compatible helper alias used by existing unit tests
    def _hidden(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self._extract_last_hidden_state(model, input_ids, attention_mask, position_ids)

    def _project_hidden_chunk(self, model: nn.Module, hidden_chunk: torch.Tensor) -> torch.Tensor:
        """Project a ``[B, chunk, H]`` slice through ``lm_head`` with FSDP2 / ZeRO-3 unshard safety."""
        core = _core_pretrained_model(model)
        if hasattr(core, "logits") and callable(core.logits):
            lm_head = getattr(core, "lm_head", None)
            if lm_head is not None and _needs_fsdp_or_zero_redirect(lm_head):
                weight = lm_head.weight
                return _forward_redirect(model, lm_head, hidden_chunk.to(weight.dtype))
            return core.logits(hidden_chunk)

        lm_head = _locate_lm_head(model)
        if lm_head is None:
            raise TypeError(f"Cannot locate `lm_head` / output embeddings on {type(core).__name__}")
        if _needs_fsdp_or_zero_redirect(lm_head):
            return _forward_redirect(model, lm_head, hidden_chunk)
        weight = getattr(lm_head, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.is_floating_point() and hidden_chunk.dtype != weight.dtype:
            hidden_chunk = hidden_chunk.to(weight.dtype)
        return lm_head(hidden_chunk)

    def _logits(self, model: nn.Module, hidden_chunk: torch.Tensor) -> torch.Tensor:
        return self._project_hidden_chunk(model, hidden_chunk)

    def compute_loss(self, model: nn.Module, inputs: dict[str, Any], return_outputs: bool = False, **kwargs):
        input_ids: torch.Tensor = inputs["input_ids"]
        attention_mask: Optional[torch.Tensor] = inputs.get("attention_mask")
        position_ids: Optional[torch.Tensor] = inputs.get("position_ids")
        labels: Optional[torch.Tensor] = inputs.get("labels")
        loss_mask: Optional[torch.Tensor] = inputs.get("loss_mask")

        if loss_mask is None:
            if attention_mask is not None:
                loss_mask = attention_mask.to(dtype=torch.float32)
            else:
                loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
        else:
            loss_mask = loss_mask.to(device=input_ids.device, dtype=torch.float32)

        need_ce = self.distill.loss == "ce" or self.distill.ce_weight > 0.0
        need_kd = self.distill.loss != "ce" and self.distill.kd_weight > 0.0
        if need_ce and labels is None:
            raise ValueError("ce_weight > 0 requires labels for the CE term (or loss='ce' requires labels).")

        offline_vals: Optional[torch.Tensor] = inputs.get("teacher_topk_values")
        offline_idx: Optional[torch.Tensor] = inputs.get("teacher_topk_indices")
        has_offline_topk = offline_vals is not None and offline_idx is not None

        teacher_hidden: Optional[torch.Tensor] = None
        if need_kd and not has_offline_topk:
            if self.self_teacher:
                with _self_teacher_context(model):
                    teacher_hidden = self._extract_last_hidden_state(model, input_ids, attention_mask, position_ids)
            elif self.teacher is not None:
                with torch.no_grad():
                    if self.distill.teacher_device == "cpu":
                        t_ids = input_ids.cpu()
                        t_attn = attention_mask.cpu() if attention_mask is not None else None
                        t_pos = position_ids.cpu() if position_ids is not None else None
                    else:
                        t_ids, t_attn, t_pos = input_ids, attention_mask, position_ids
                    teacher_hidden = self._extract_last_hidden_state(self.teacher, t_ids, t_attn, t_pos)
            else:
                raise ValueError(
                    "Online KD requires `teacher_model` (`nn.Module` or `'self'`), or offline "
                    "`teacher_topk_values` and `teacher_topk_indices` in batch inputs."
                )

        student_hidden = self._extract_last_hidden_state(model, input_ids, attention_mask, position_ids)

        seq_len = input_ids.shape[1]
        chunk_size = self.distill.chunk_size or seq_len
        kd_local_sum = input_ids.new_zeros((), dtype=torch.float32)
        kd_local_tokens = loss_mask.sum().float() if need_kd else input_ids.new_zeros((), dtype=torch.float32)
        ce_local_sum = input_ids.new_zeros((), dtype=torch.float32)
        ce_local_tokens = input_ids.new_zeros((), dtype=torch.float32)

        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            student_logits = self._project_hidden_chunk(model, student_hidden[:, start:end])

            if need_kd:
                mask_chunk = loss_mask[:, start:end]
                if has_offline_topk:
                    assert offline_vals is not None and offline_idx is not None
                    t_vals = offline_vals[:, start:end].to(device=student_logits.device)
                    t_idx = offline_idx[:, start:end].to(device=student_logits.device, dtype=torch.long)
                    s_vals = torch.gather(student_logits, dim=-1, index=t_idx)
                    if self.distill.reverse:
                        s_vals, t_vals = t_vals, s_vals
                    token_kd = topk_kl_from_topk(s_vals, t_vals, temperature=self.distill.temperature)
                else:
                    with torch.no_grad():
                        assert teacher_hidden is not None
                        teacher_ref = model if self.self_teacher else self.teacher
                        assert teacher_ref is not None
                        if self.self_teacher:
                            with _self_teacher_context(model):
                                teacher_logits = self._project_hidden_chunk(model, teacher_hidden[:, start:end])
                        else:
                            teacher_logits = self._project_hidden_chunk(teacher_ref, teacher_hidden[:, start:end])
                        teacher_logits = teacher_logits.to(device=student_logits.device)

                    if self.distill.loss == "kl":
                        token_kd = full_vocab_kl(
                            student_logits,
                            teacher_logits,
                            temperature=self.distill.temperature,
                            reverse=self.distill.reverse,
                        )
                    else:
                        eff_k = min(self.distill.top_k, student_logits.shape[-1])
                        token_kd = topk_kl(
                            student_logits,
                            teacher_logits,
                            k=eff_k,
                            temperature=self.distill.temperature,
                            reverse=self.distill.reverse,
                        )
                kd_local_sum = kd_local_sum + (token_kd * mask_chunk).sum()

            if need_ce and labels is not None:
                ce_end = min(end, seq_len - 1)
                if ce_end > start:
                    ce_logits = student_logits[:, : ce_end - start, :].reshape(-1, student_logits.shape[-1]).float()
                    ce_targets = labels[:, start + 1 : ce_end + 1].to(device=student_logits.device).reshape(-1)
                    ce_local_sum = ce_local_sum + F.cross_entropy(
                        ce_logits, ce_targets, ignore_index=-100, reduction="sum"
                    )
                    ce_local_tokens = ce_local_tokens + (ce_targets != -100).sum().float()

        is_dist = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if is_dist else 1

        total_loss = input_ids.new_zeros((), dtype=torch.float32)
        kd_logged = 0.0
        kd_tokens_logged = 0.0
        ce_logged = 0.0

        if need_kd:
            global_kd_tokens = kd_local_tokens.clone()
            if is_dist:
                dist.all_reduce(global_kd_tokens, op=dist.ReduceOp.SUM)
            denom = global_kd_tokens.clamp(min=1.0)
            kd_term = (world_size * kd_local_sum / denom) * self.distill.kd_weight
            total_loss = total_loss + kd_term
            kd_logged = float(kd_local_sum.detach()) / max(float(kd_local_tokens.detach()), 1.0)
            kd_tokens_logged = float(global_kd_tokens.detach())

        if need_ce:
            global_ce_tokens = ce_local_tokens.clone()
            if is_dist:
                dist.all_reduce(global_ce_tokens, op=dist.ReduceOp.SUM)
            ce_denom = global_ce_tokens.clamp(min=1.0)
            ce_term = (world_size * ce_local_sum / ce_denom) * self.distill.ce_weight
            total_loss = total_loss + ce_term
            ce_logged = float(ce_local_sum.detach()) / max(float(ce_local_tokens.detach()), 1.0)

        self._last_loss_parts = {
            "kd_loss": kd_logged,
            "kd_tokens": kd_tokens_logged,
            "ce_loss": ce_logged,
        }
        return (total_loss, None) if return_outputs else total_loss


__all__ = ["KDTrainer"]
