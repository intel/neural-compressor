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
"""Declarative Knowledge Distillation configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any, Optional

import torch.distributed as dist
from transformers import TrainingArguments

_VALID_LOSSES = ("topk_kl", "kl", "ce")
_VALID_TEACHER_DEVICES = ("auto", "shard", "sharded", "local", "cuda", "cpu", "self", "offline")


@dataclass
class DistillationConfig:
    """Teacher placement and divergence loss configuration for :class:`KDTrainer` and :class:`QADTrainer`.

    Attributes:
        teacher: Teacher model specification: ``None``, ``"self"`` (zero-VRAM LoRA self-teacher),
            a HuggingFace model path/ID string, or a pre-loaded ``torch.nn.Module``.
        loss: ``"topk_kl"`` (default, ModelOpt-aligned), ``"kl"`` (full-vocab KL), or ``"ce"``.
        top_k: Number of highest-logit teacher vocabulary entries retained per token.
        temperature: Logit temperature (applied without any ``T^2`` multiplier, matching ModelOpt).
        reverse: Swap ``(student_topk, teacher_topk)`` *after* gathering on teacher Top-K indices.
        kd_weight: Scalar multiplier on the knowledge-distillation term.
        ce_weight: Scalar multiplier on the next-token Cross-Entropy term (requires ``labels``).
        chunk_size: Sequence token chunk size for memory-bounded ``lm_head`` projection (``0`` = unchunked).
        teacher_device: ``"shard"`` (FSDP2 sharded), ``"cuda"`` (replicated GPU), ``"cpu"`` (offloaded),
            ``"self"`` (in-place adapter/fake-quant toggle), or ``"offline"`` (precomputed Top-K shards).
        offline_topk_dir: Optional directory containing precomputed teacher Top-K shards.
    """

    teacher: Optional[Any] = None
    loss: str = "topk_kl"
    top_k: int = 4096
    temperature: float = 1.0
    reverse: bool = False
    kd_weight: float = 1.0
    ce_weight: float = 0.0
    chunk_size: int = 1024
    teacher_device: str = "shard"
    offline_topk_dir: Optional[str] = None

    def __init__(
        self,
        teacher: Optional[Any] = None,
        loss: str = "topk_kl",
        top_k: int = 4096,
        temperature: float = 1.0,
        reverse: bool = False,
        kd_weight: float = 1.0,
        ce_weight: float = 0.0,
        chunk_size: int = 1024,
        teacher_device: str = "shard",
        offline_topk_dir: Optional[str] = None,
        *,
        chunk: Optional[int] = None,
    ) -> None:
        self.teacher = teacher
        self.loss = loss
        self.top_k = top_k
        self.temperature = temperature
        self.reverse = reverse
        self.kd_weight = kd_weight
        self.ce_weight = ce_weight
        self.chunk_size = chunk if chunk is not None else chunk_size
        self.teacher_device = "self" if teacher == "self" and teacher_device == "shard" else teacher_device
        self.offline_topk_dir = offline_topk_dir
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.loss not in _VALID_LOSSES:
            raise ValueError(f"unknown distill loss {self.loss!r}; expected one of {_VALID_LOSSES}")
        if self.teacher_device not in _VALID_TEACHER_DEVICES:
            raise ValueError(
                f"unknown teacher_device {self.teacher_device!r}; expected one of {_VALID_TEACHER_DEVICES}"
            )
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.chunk_size < 0:
            raise ValueError("chunk_size must be non-negative")
        if self.loss == "ce" and (self.kd_weight != 0 or self.ce_weight == 0):
            raise ValueError("plain CE requires kd_weight=0 and a non-zero ce_weight")
        if self.loss != "ce" and self.kd_weight == 0 and self.ce_weight == 0:
            raise ValueError("at least one of kd_weight/ce_weight must be non-zero")

    @property
    def chunk(self) -> int:
        """Backward-compatible alias for :attr:`chunk_size`."""
        return self.chunk_size

    @chunk.setter
    def chunk(self, value: int) -> None:
        if value < 0:
            raise ValueError("chunk_size must be non-negative")
        self.chunk_size = value

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "teacher": self.teacher if not hasattr(self.teacher, "parameters") else None,
            "loss": self.loss,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "reverse": self.reverse,
            "kd_weight": self.kd_weight,
            "ce_weight": self.ce_weight,
            "chunk_size": self.chunk_size,
            "teacher_device": self.teacher_device,
        }
        if self.offline_topk_dir is not None:
            payload["offline_topk_dir"] = self.offline_topk_dir
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DistillationConfig":
        data = dict(payload)
        if "chunk" in data and "chunk_size" not in data:
            data["chunk_size"] = data.pop("chunk")
        else:
            data.pop("chunk", None)
        return cls(**data)


# Backward-compatible alias
DistillConfig = DistillationConfig


@dataclass
class DistillationTrainingArguments(TrainingArguments):
    """``transformers.TrainingArguments`` extended with knowledge-distillation CLI options."""

    teacher_model_name_or_path: Optional[str] = field(default=None)
    teacher_placement: str = field(default="shard")
    kd_loss_type: str = field(default="topk_kl")
    kd_top_k: int = field(default=4096)
    kd_temperature: float = field(default=1.0)
    kd_reverse_kl: bool = field(default=False)
    kd_loss_weight: float = field(default=1.0)
    ce_loss_weight: float = field(default=0.0)
    logits_chunk_size: int = field(default=1024)

    def __post_init__(self) -> None:
        fsdp_str = str(getattr(self, "fsdp", "") or "")
        fsdp_cfg = getattr(self, "fsdp_config", None)
        needs_cpu_pg = ("offload" in fsdp_str) or (
            isinstance(fsdp_cfg, dict)
            and (
                bool(fsdp_cfg.get("cpu_offload", False))
                or str(fsdp_cfg.get("state_dict_type", "")) == "FULL_STATE_DICT"
            )
        )
        saved_env: dict[str, Optional[str]] = {}
        if needs_cpu_pg and not dist.is_initialized():
            for key, val in (("ACCELERATE_USE_FSDP", "true"), ("FSDP_OFFLOAD_PARAMS", "true")):
                saved_env[key] = os.environ.get(key)
                if saved_env[key] is None:
                    os.environ[key] = val
        try:
            super().__post_init__()
            if saved_env:
                _ = self.device
        finally:
            for key, prev in saved_env.items():
                if prev is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = prev


__all__ = ["DistillConfig", "DistillationConfig", "DistillationTrainingArguments"]
