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
"""Knowledge-distillation losses (teacher/student logits)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def teacher_topk(teacher_logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the teacher's top-k values and indices along the vocabulary axis."""
    if k > teacher_logits.shape[-1]:
        raise ValueError(f"top_k={k} > vocab={teacher_logits.shape[-1]}")
    return torch.topk(teacher_logits, k=k, dim=-1)


def topk_kl_from_topk(
    student_topk: torch.Tensor,
    teacher_topk_values: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Per-token KL over the selected top-k entries.

    ``F.kl_div(input=student, target=teacher, log_target=True)`` uses the same
    argument order as ModelOpt's ``TopKLogitsKLLoss``. There is no ``T^2``
    multiplier: the temperature only rescales the logits.
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
    """Top-k KL between full logits; ``reverse`` swaps teacher/student."""
    if reverse:
        student_logits, teacher_logits = teacher_logits, student_logits
    teacher_values, teacher_indices = teacher_topk(teacher_logits, k)
    student_values = torch.gather(student_logits, dim=-1, index=teacher_indices)
    return topk_kl_from_topk(student_values, teacher_values, temperature=temperature)


class TopKLogitsKLLoss(nn.Module):
    """Chunked top-k KL used by :class:`QADTrainer`.

    ``forward`` accepts full logits and an optional token mask; it slices the
    sequence into ``chunk`` tokens so the ``[batch, seq, vocab]`` intermediate
    never materializes for 150k-vocab models. ``teacher`` may also be a
    precomputed ``(values, indices)`` pair to save memory.

    Args:
        top_k: Number of teacher vocabulary entries kept.
        temperature: Logit temperature.
        chunk: Token chunk size (0 disables chunking).
        reverse: Compute ``KL(teacher || student)`` instead.
    """

    def __init__(self, top_k: int = 4096, temperature: float = 1.0, chunk: int = 1024, reverse: bool = False):
        super().__init__()
        self.top_k = top_k
        self.temperature = temperature
        self.chunk = chunk
        self.reverse = reverse

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        loss_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(summed_loss, token_count)`` over the sequence.

        Chunking along ``dim=1`` keeps semantics identical to a full-sequence
        computation: each token's loss only depends on its own logits.
        """
        if self.reverse and isinstance(teacher, torch.Tensor):
            student_logits, teacher = teacher, student_logits
        seq = student_logits.shape[1]
        chunk = self.chunk or seq
        total = student_logits.new_zeros((), dtype=torch.float32)
        count = student_logits.new_zeros((), dtype=torch.float32)
        for start in range(0, seq, chunk):
            end = min(start + chunk, seq)
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

    def _chunk_loss(self, student_logits, teacher) -> torch.Tensor:
        if isinstance(teacher, tuple):
            values, indices = teacher
            student_values = torch.gather(student_logits, dim=-1, index=indices)
            return topk_kl_from_topk(student_values, values, temperature=self.temperature)
        return topk_kl(
            student_logits,
            teacher,
            k=self.top_k,
            temperature=self.temperature,
            reverse=False,
        )

    def extra_repr(self) -> str:
        """Show the loss configuration."""
        return f"top_k={self.top_k}, temperature={self.temperature}, chunk={self.chunk}"


__all__ = ["TopKLogitsKLLoss", "teacher_topk", "topk_kl", "topk_kl_from_topk"]
