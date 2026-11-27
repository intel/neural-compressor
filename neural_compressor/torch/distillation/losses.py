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
"""ModelOpt-aligned Top-K and full-vocabulary KL distillation losses."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def teacher_topk(teacher_logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the teacher's top-k values and vocabulary indices along ``dim=-1``."""
    if k > teacher_logits.shape[-1]:
        raise ValueError(f"top_k={k} > vocab={teacher_logits.shape[-1]}")
    return torch.topk(teacher_logits, k=k, dim=-1)


def topk_kl_from_topk(
    student_topk: torch.Tensor,
    teacher_topk_values: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute per-token KL divergence over pre-gathered Top-K logits in FP32.

    Matches NVIDIA ``ModelOpt``'s ``TopKLogitsKLLoss`` exactly:
    ``F.kl_div(log_softmax(s / T), log_softmax(t / T), log_target=True).sum(-1)``
    with no ``T^2`` scaling factor.
    """
    student_log_probs = F.log_softmax(student_topk.float() / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_topk_values.float() / temperature, dim=-1)
    return F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True).sum(dim=-1)


def topk_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    k: int = 4096,
    temperature: float = 1.0,
    reverse: bool = False,
) -> torch.Tensor:
    """Compute per-token Top-K KL; ``reverse=True`` swaps distributions *after* teacher Top-K selection."""
    teacher_values, teacher_indices = teacher_topk(teacher_logits, k)
    student_values = torch.gather(student_logits, dim=-1, index=teacher_indices)
    if reverse:
        student_values, teacher_values = teacher_values, student_values
    return topk_kl_from_topk(student_values, teacher_values, temperature=temperature)


def full_vocab_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    reverse: bool = False,
) -> torch.Tensor:
    """Compute per-token full-vocabulary KL divergence in FP32."""
    student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
    if reverse:
        student_log_probs, teacher_log_probs = teacher_log_probs, student_log_probs
    return F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True).sum(dim=-1)


class TopKLogitsKLLoss(nn.Module):
    """Sequence-chunked Top-K KL loss supporting both online teacher logits and precomputed ``(values, indices)``."""

    def __init__(
        self,
        top_k: int = 4096,
        temperature: float = 1.0,
        chunk_size: int = 1024,
        reverse: bool = False,
        *,
        chunk: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.temperature = temperature
        self.chunk_size = chunk if chunk is not None else chunk_size
        self.reverse = reverse

    @property
    def chunk(self) -> int:
        return self.chunk_size

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        loss_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(masked_loss_sum, valid_token_count)`` in ``float32``."""
        seq_len = student_logits.shape[1]
        step = self.chunk_size or seq_len
        total = student_logits.new_zeros((), dtype=torch.float32)
        count = student_logits.new_zeros((), dtype=torch.float32)

        for start in range(0, seq_len, step):
            end = min(start + step, seq_len)
            teacher_chunk = (
                (teacher[0][:, start:end], teacher[1][:, start:end])
                if isinstance(teacher, tuple)
                else teacher[:, start:end]
            )
            token_loss = self._chunk_loss(student_logits[:, start:end], teacher_chunk)
            if loss_mask is not None:
                mask = loss_mask[:, start:end].float()
                total = total + (token_loss * mask).sum()
                count = count + mask.sum()
            else:
                total = total + token_loss.sum()
                count = count + token_loss.numel()
        return total, count

    def _chunk_loss(
        self,
        student_logits: torch.Tensor,
        teacher: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if isinstance(teacher, tuple):
            values, indices = teacher
            if indices.dtype != torch.long:
                indices = indices.long()
            student_values = torch.gather(student_logits, dim=-1, index=indices)
            if self.reverse:
                student_values, values = values, student_values
            return topk_kl_from_topk(student_values, values, temperature=self.temperature)
        return topk_kl(
            student_logits,
            teacher,
            k=self.top_k,
            temperature=self.temperature,
            reverse=self.reverse,
        )

    def extra_repr(self) -> str:
        return (
            f"top_k={self.top_k}, temperature={self.temperature}, "
            f"chunk_size={self.chunk_size}, reverse={self.reverse}"
        )


class LogitsKLLoss(nn.Module):
    """Full-vocabulary KL divergence loss module with optional temperature scaling and reverse KL."""

    def __init__(
        self,
        temperature: float = 1.0,
        reverse: bool = False,
        scale_by_temperature_sq: bool = False,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.reverse = reverse
        self.scale_by_temperature_sq = scale_by_temperature_sq

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        loss = full_vocab_kl(
            student_logits,
            teacher_logits,
            temperature=self.temperature,
            reverse=self.reverse,
        )
        if self.scale_by_temperature_sq:
            loss = loss * (self.temperature**2)
        return loss


__all__ = [
    "LogitsKLLoss",
    "TopKLogitsKLLoss",
    "full_vocab_kl",
    "teacher_topk",
    "topk_kl",
    "topk_kl_from_topk",
]
