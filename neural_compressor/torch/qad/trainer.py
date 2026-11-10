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
"""QADTrainer: HuggingFace ``Trainer`` + teacher top-k KL.

The trainer owns the distillation loss (not a wrapped model) so FSDP2,
PEFT and ``Trainer`` checkpointing keep working unchanged. Quantization is
inserted beforehand by ``quantize(model, FakeQuantConfig, ...)``.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import Trainer

from neural_compressor.torch.distill import DistillConfig, topk_kl
from neural_compressor.torch.quantization.fake_quant import FakeQuantConfig

from .checkpoint import is_peft_model, save_adapter, save_full_state_dict
from .metrics import memory_report, param_counts, tflops_and_mfu
from .modeling import get_adapter, logits_from_hidden


def _unwrap(model: Any) -> Any:
    return getattr(model, "module", model)


def _find_attr(model: Any, name: str, seen: Optional[set] = None):
    """Walk wrapper attributes (PeftModel/DDP) looking for a method."""
    if seen is None:
        seen = set()
    if id(model) in seen:
        return None
    seen.add(id(model))
    if hasattr(model, name):
        return getattr(model, name)
    for attr in ("module", "base_model", "model", "base"):
        sub = getattr(model, attr, None)
        if sub is not None:
            found = _find_attr(sub, name, seen)
            if found is not None:
                return found
    return None


def _chunked_kd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    top_k: int,
    temperature: float,
    reverse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k KL for one chunk; returns ``(masked_sum, masked_tokens)``."""
    token_loss = topk_kl(student_logits, teacher_logits, k=top_k, temperature=temperature, reverse=reverse)
    mask = loss_mask.float()
    return (token_loss * mask).sum(), mask.sum()


class QADTrainer(Trainer):
    """Trainer that computes ``kd_weight * KD + ce_weight * CE``.

    Args:
        teacher: Loaded teacher (same layout as the student).
        distill_config: Teacher/loss settings; defaults to ``qad_config.distill``.
        qad_config: Optional full recipe (used for metadata and defaults).
        quant_config: FakeQuantConfig used on the student; written next to
            checkpoints so the quantizers can be rebuilt.
        mode: ``"full"`` or ``"lora"`` (affects the FLOPs estimate only).
        backend: Label for logging (``"hf"``/``"unsloth"``).
    """

    def __init__(
        self,
        *args,
        teacher: torch.nn.Module,
        distill_config: Optional[DistillConfig] = None,
        qad_config=None,
        quant_config: Optional[FakeQuantConfig] = None,
        mode: str = "full",
        backend: str = "hf",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if distill_config is None and qad_config is not None:
            distill_config = qad_config.distill
        if distill_config is None:
            raise ValueError("QADTrainer needs a DistillConfig (or a QADConfig)")
        self.teacher = teacher
        self.teacher.eval()
        self.teacher.requires_grad_(False)
        self.distill = distill_config
        self.qad_config = qad_config
        self.quant_config = quant_config
        self.mode = mode
        self.backend = backend
        self.adapter = get_adapter(self.model)
        self.teacher_adapter = get_adapter(teacher)
        self._last_log_time = time.time()
        self._last_log_step = 0
        self._last_loss_parts: dict[str, float] = {}
        self._peak_reset = False

    # --------------------------------------------------------------- training
    def training_step(self, *args, **kwargs):
        """Run one step and reset the peak-memory window on the first one."""
        if os.environ.get("NC_QAD_DEBUG_MEMORY") == "1":
            report = memory_report(self.args.device)
            print(
                f"mem[step_entry] global_step={self.state.global_step} "
                f"alloc={report['allocated'] / 2**30:.1f}GiB peak={report['peak'] / 2**30:.1f}GiB",
                flush=True,
            )
        # peak_gib should describe the step, not the loader or FSDP2 wrapping.
        if not self._peak_reset:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            self._peak_reset = True
        if os.environ.get("NC_QAD_FORCE_AUTOCAST") == "1" and torch.cuda.is_available():
            # HF's FSDP2 path runs without autocast while the plain bf16 path
            # uses it; this switch restores the non-FSDP numerics for A/B.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return super().training_step(*args, **kwargs)
        return super().training_step(*args, **kwargs)

    # ------------------------------------------------------------------- loss
    def _hidden(self, model: torch.nn.Module, input_ids, attention_mask):
        module = _unwrap(model)
        if hasattr(module, "forward") and hasattr(module, "forward_hidden"):
            # Execute through the original wrapper so DDP/PEFT/FSDP hooks and
            # gradient reduction keep their required semantics.
            return model(input_ids=input_ids, attention_mask=attention_mask)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = getattr(outputs, "hidden_states", None)
        if hidden is None:
            raise TypeError(f"{type(module).__name__} did not return hidden_states; use the CausalLM interface")
        return hidden[-1]

    def _logits(self, model: torch.nn.Module, hidden: torch.Tensor):
        module = _unwrap(model)
        fn = _find_attr(module, "logits")
        if fn is not None:
            return fn(hidden)
        return logits_from_hidden(module, hidden, self.adapter)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Chunk the sequence, compute top-k KL, and reduce tokens globally."""
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        loss_mask = inputs.get("loss_mask")
        if loss_mask is None:
            loss_mask = torch.ones_like(input_ids, dtype=torch.float32)

        if self.distill.loss == "ce":
            if "labels" not in inputs:
                raise ValueError("loss='ce' requires labels")
            loss = self.distill.ce_weight * self._ce_loss(model, inputs)
            self._last_loss_parts = {"kd_loss": 0.0, "kd_tokens": 0.0}
            return (loss, None) if return_outputs else loss

        if self.distill.ce_weight and "labels" not in inputs:
            raise ValueError("ce_weight > 0 requires labels for the CE term")

        student_hidden = self._hidden(model, input_ids, attention_mask)
        with torch.no_grad():
            if self.distill.teacher_device == "cpu":
                teacher_input_ids = input_ids.cpu()
                teacher_attention_mask = attention_mask.cpu() if attention_mask is not None else None
            else:
                teacher_input_ids, teacher_attention_mask = input_ids, attention_mask
            teacher_hidden = self._hidden(self.teacher, teacher_input_ids, teacher_attention_mask)

        local_sum = input_ids.new_zeros((), dtype=torch.float32)
        local_tokens = loss_mask.sum().float()
        chunk = self.distill.chunk or input_ids.shape[1]
        for start in range(0, input_ids.shape[1], chunk):
            end = min(start + chunk, input_ids.shape[1])
            with torch.no_grad():
                teacher_logits = self._logits(self.teacher, teacher_hidden[:, start:end])
            student_logits = self._logits(model, student_hidden[:, start:end])
            teacher_logits = teacher_logits.to(student_logits.device)
            if self.distill.loss == "kl":
                student_log_probs = F.log_softmax(student_logits.float() / self.distill.temperature, dim=-1)
                teacher_log_probs = F.log_softmax(teacher_logits.float() / self.distill.temperature, dim=-1)
                if self.distill.reverse:
                    student_log_probs, teacher_log_probs = teacher_log_probs, student_log_probs
                token_loss = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True).sum(
                    dim=-1
                )
                token_sum = (token_loss * loss_mask[:, start:end].float()).sum()
            else:
                token_sum, _ = _chunked_kd(
                    student_logits,
                    teacher_logits,
                    loss_mask[:, start:end],
                    top_k=self.distill.top_k,
                    temperature=self.distill.temperature,
                    reverse=self.distill.reverse,
                )
            local_sum = local_sum + token_sum

        if dist.is_available() and dist.is_initialized():
            world = dist.get_world_size()
            tokens = local_tokens.clone()
            dist.all_reduce(tokens, op=dist.ReduceOp.SUM)
        else:
            world, tokens = 1, local_tokens
        # FSDP2 reduces gradients with AVG; multiply by world for a global
        # token mean. Equal-length windows make this a no-op.
        loss = world * local_sum / tokens.clamp(min=1.0) * self.distill.kd_weight

        if self.distill.ce_weight and "labels" in inputs:
            loss = loss + self.distill.ce_weight * self._ce_loss(model, inputs)

        if os.environ.get("NC_QAD_DEBUG_LOSS") == "1":
            rank = dist.get_rank() if dist.is_initialized() else 0
            print(
                f"[loss] rank={rank} local_tokens={float(local_tokens):.0f} "
                f"tokens={float(tokens):.0f} local_sum={float(local_sum):.6f} "
                f"loss={float(loss):.6f}",
                flush=True,
            )
        self._last_loss_parts = {
            "kd_loss": float(local_sum.detach()) / max(float(tokens), 1.0),
            "kd_tokens": float(tokens),
        }
        return (loss, None) if return_outputs else loss

    def _ce_loss(self, model, inputs) -> torch.Tensor:
        labels = inputs["labels"]
        hidden = self._hidden(model, inputs["input_ids"], inputs.get("attention_mask"))
        logits = self._logits(model, hidden)
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
            labels[:, 1:].reshape(-1),
        )

    # ------------------------------------------------------------- checkpoint
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Save FSDP2-gathered weights plus ``quantization_config.json``."""
        out = output_dir or self.args.output_dir
        if self.is_fsdp_enabled and is_peft_model(self.model):
            if save_adapter(self.model, out, should_save=self.args.should_save):
                if self.args.should_save:
                    os.makedirs(out, exist_ok=True)
                    torch.save(self.args, os.path.join(out, "training_args.bin"))
                    self._write_quant_metadata(out)
                return
        if self.is_fsdp_enabled and not is_peft_model(self.model):
            if save_full_state_dict(self.model, out, should_save=self.args.should_save):
                if self.args.should_save:
                    torch.save(self.args, os.path.join(out, "training_args.bin"))
                    self._write_quant_metadata(out)
                return
        super().save_model(output_dir, _internal_call)
        if self.args.should_save:
            self._write_quant_metadata(out)

    def _write_quant_metadata(self, out: str) -> None:
        if self.quant_config is None:
            return
        from .export import write_quantization_config

        write_quantization_config(_unwrap(self.model), out, self.quant_config)

    # -------------------------------------------------------------------- log
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """Add throughput, FLOPs, memory and parameter counts to Trainer logs."""
        now = time.time()
        step = int(self.state.global_step)
        delta = step - self._last_log_step
        world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        device = self.args.device
        device_name = ""
        try:
            if torch.cuda.is_available() and torch.device(device).type == "cuda":
                device_name = torch.cuda.get_device_name(torch.device(device))
        except Exception:
            pass
        if delta > 0:
            seq = self.qad_config.train.seq_length if self.qad_config else 0
            tokens = delta * self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps * world * seq
            seconds = max(now - self._last_log_time, 1e-9)
            counts = param_counts(_unwrap(self.model))
            teacher_params = sum(p.numel() for p in _unwrap(self.teacher).parameters())
            perf = tflops_and_mfu(
                tokens=tokens,
                step_seconds=seconds,
                params=counts["params_total"],
                mode=self.mode,
                teacher_params=teacher_params,
                device_name=device_name,
                world_size=world,
                recompute=bool(
                    self.args.gradient_checkpointing
                    or (self.args.fsdp_config or {}).get("activation_checkpointing", False)
                ),
            )
            logs.update(
                {
                    "tokens": float(tokens),
                    "tokens_per_s": tokens / seconds,
                    "step_s": seconds / delta,
                    "backend": self.backend,
                    **counts,
                    **{key: value for key, value in perf.items() if value is not None},
                }
            )
        memory = memory_report(device)
        logs["peak_gib"] = memory["peak"] / (1024**3)
        logs["alloc_gib"] = memory["allocated"] / (1024**3)
        logs.update(self._last_loss_parts)
        self._last_log_time, self._last_log_step = now, step
        super().log(logs, start_time=start_time)


__all__ = ["QADTrainer"]
