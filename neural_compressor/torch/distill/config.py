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
"""Distillation configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class DistillConfig:
    """Teacher and loss settings for knowledge distillation.

    Attributes:
        teacher: HF id/path or an already loaded model. ``None`` means "same as
            the student path" (self-distillation on a quantized student).
        loss: ``"topk_kl"`` (default), ``"kl"`` or ``"ce"``.
        top_k: Teacher vocabulary entries kept by the top-k KL.
        temperature: Logit temperature; no ``T^2`` factor is applied.
        reverse: Compute ``KL(teacher || student)``.
        kd_weight: Weight of the distillation term.
        ce_weight: Weight of the cross-entropy term (labels required).
        chunk: Token chunk size used to bound the logits memory.
        teacher_device: ``"shard"`` (FSDP2 layer sharding), ``"cuda"`` (same
            device) or ``"cpu"`` (offload).
    """

    teacher: Optional[Any] = None
    loss: str = "topk_kl"
    top_k: int = 4096
    temperature: float = 1.0
    reverse: bool = False
    kd_weight: float = 1.0
    ce_weight: float = 0.0
    chunk: int = 1024
    teacher_device: str = "shard"

    def __post_init__(self) -> None:
        """Validate the loss and device policy."""
        if self.loss not in ("topk_kl", "kl", "ce"):
            raise ValueError(f"unknown distill loss {self.loss!r}")
        if self.teacher_device not in ("shard", "cuda", "cpu"):
            raise ValueError(f"unknown teacher_device {self.teacher_device!r}")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.chunk < 0:
            raise ValueError("chunk must be non-negative")
        if self.loss == "ce" and (self.kd_weight != 0 or self.ce_weight == 0):
            raise ValueError("plain CE requires kd_weight=0 and a non-zero ce_weight")
        if self.loss != "ce" and self.kd_weight == 0 and self.ce_weight == 0:
            raise ValueError("at least one of kd_weight/ce_weight must be non-zero")

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dictionary (teacher kept as-is)."""
        return {
            "teacher": self.teacher,
            "loss": self.loss,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "reverse": self.reverse,
            "kd_weight": self.kd_weight,
            "ce_weight": self.ce_weight,
            "chunk": self.chunk,
            "teacher_device": self.teacher_device,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "DistillConfig":
        """Build from :meth:`to_dict` output."""
        return cls(**payload)


__all__ = ["DistillConfig"]
