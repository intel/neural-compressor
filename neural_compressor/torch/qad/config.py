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
"""QAD configuration: model + quantization + distillation + training + export."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from neural_compressor.torch.distill import DistillConfig
from neural_compressor.torch.quantization.fake_quant import FakeQuantConfig


@dataclass
class ModelConfig:
    """Which checkpoint to train and how to load it."""

    path: str
    family: Optional[str] = None
    trust_remote_code: bool = False
    dtype: str = "bfloat16"
    tiny: bool = False

    def to_dict(self) -> dict:
        """Serialize to a dictionary."""
        return {
            "path": self.path,
            "family": self.family,
            "trust_remote_code": self.trust_remote_code,
            "dtype": self.dtype,
            "tiny": self.tiny,
        }


@dataclass
class TrainConfig:
    """Training-loop settings mirrored by the example's CLI."""

    backend: str = "hf"
    parallel: str = "none"  # "none" | "fsdp2"
    seq_length: int = 8192
    micro_batch_size: int = 1
    global_batch_size: int = 16
    train_iters: int = 10
    lr: float = 2e-4
    min_lr: float = 2e-5
    warmup: int = 5
    weight_decay: float = 0.0
    clip_grad: float = 1.0
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    seed: int = 42
    gradient_checkpointing: bool = False
    gc_mode: str = "trainer"
    save_steps: int = 0
    logging_steps: int = 1
    output_dir: str = ""
    lora: Optional[dict] = None

    def __post_init__(self) -> None:
        """Validate batch arithmetic and enums."""
        if self.backend not in ("hf", "unsloth"):
            raise ValueError(f"unknown backend {self.backend!r}")
        if self.parallel not in ("none", "fsdp2"):
            raise ValueError(f"unknown parallel mode {self.parallel!r}")
        if self.micro_batch_size <= 0 or self.global_batch_size <= 0 or self.seq_length <= 0:
            raise ValueError("seq_length and batch sizes must be positive")
        if self.global_batch_size % self.micro_batch_size:
            raise ValueError("global_batch_size must be divisible by micro_batch_size")
        if self.train_iters <= 0 or self.lr <= 0 or self.min_lr < 0 or self.warmup < 0:
            raise ValueError("train_iters/lr must be positive and min_lr/warmup non-negative")
        if self.gc_mode not in ("trainer", "fsdp"):
            raise ValueError(f"unknown gc_mode {self.gc_mode!r}")

    @property
    def grad_accum(self) -> int:
        """Gradient accumulation steps."""
        return self.global_batch_size // self.micro_batch_size

    def to_dict(self) -> dict:
        """Serialize to a dictionary."""
        return dict(self.__dict__)


@dataclass
class ExportConfig:
    """Export format and layout."""

    format: str = "hf_bf16"
    keep_vision: bool = True
    lora: str = "merge"
    max_shard_size: str = "10GB"

    def __post_init__(self) -> None:
        """Validate the format."""
        if self.format not in ("hf_bf16",):
            raise ValueError(
                f"unsupported export format {self.format!r}; "
                "real MXFP4/FP8 export is tracked as a follow-up (use ModelOpt/llm-compressor)"
            )
        if self.lora not in ("merge", "adapter"):
            raise ValueError(f"unsupported LoRA export mode {self.lora!r}")

    def to_dict(self) -> dict:
        """Serialize to a dictionary."""
        return dict(self.__dict__)


@dataclass
class QADConfig:
    """Top-level recipe: what to quantize, distill from, and how to train."""

    model: ModelConfig
    quant: FakeQuantConfig
    distill: DistillConfig
    train: TrainConfig = field(default_factory=TrainConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    @classmethod
    def from_dict(cls, payload: dict) -> "QADConfig":
        """Build a config from a recipe dictionary."""
        quant = payload.get("quant") or {}
        quant = quant if isinstance(quant, FakeQuantConfig) else FakeQuantConfig.from_dict(quant)
        distill = payload.get("distill") or {}
        distill = distill if isinstance(distill, DistillConfig) else DistillConfig.from_dict(distill)
        model = payload.get("model") or {}
        train = payload.get("train") or {}
        export = payload.get("export") or {}
        return cls(
            model=ModelConfig(**model),
            quant=quant,
            distill=distill,
            train=TrainConfig(**train),
            export=ExportConfig(**export),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "QADConfig":
        """Load a recipe from YAML (``schema: neural_compressor.qad/v1``)."""
        import yaml

        payload = yaml.safe_load(Path(path).read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: recipe must be a mapping")
        schema = payload.get("schema", "neural_compressor.qad/v1")
        if schema != "neural_compressor.qad/v1":
            raise ValueError(f"{path}: unsupported recipe schema {schema!r}")
        return cls.from_dict(payload)

    def to_dict(self) -> dict:
        """Serialize to a recipe dictionary."""
        return {
            "schema": "neural_compressor.qad/v1",
            "model": self.model.to_dict(),
            "quant": self.quant.to_dict(),
            "distill": self.distill.to_dict(),
            "train": self.train.to_dict(),
            "export": self.export.to_dict(),
        }


__all__ = ["ExportConfig", "ModelConfig", "QADConfig", "TrainConfig"]
