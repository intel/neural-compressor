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
"""QADTrainingArguments combining QuantizationTrainingArguments and DistillationTrainingArguments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from neural_compressor.torch.distillation.config import DistillationConfig, DistillationTrainingArguments
from neural_compressor.torch.quantization.fake_quant import FakeQuantConfig
from neural_compressor.torch.quantization.qat_trainer import QuantizationTrainingArguments


@dataclass
class QADTrainingArguments(DistillationTrainingArguments, QuantizationTrainingArguments):
    """Unified HuggingFace ``TrainingArguments`` for Quantization-Aware Distillation (QAD).

    Inherits all ``transformers.TrainingArguments`` flags (FSDP2, DeepSpeed, gradient
    checkpointing, ``bf16``, ``lr_scheduler_type``, etc.) plus quantization and distillation
    CLI options so standard ``HfArgumentParser`` YAML/CLI workflows work out of the box.
    """

    seq_length: int = field(
        default=8192,
        metadata={"help": "Sequence length per sample used for packing/dataset loading and throughput reporting."},
    )
    export_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Optional directory to export inference checkpoint after training."},
    )
    export_format: str = field(
        default="compressed_tensors",
        metadata={"help": "Inference export format ('compressed_tensors' or 'bf16_qdq')."},
    )
    stop_after_steps: int = field(
        default=0,
        metadata={"help": "Optional early stop step count while keeping full max_steps LR schedule."},
    )
    grad_debug: bool = field(
        default=False,
        metadata={"help": "Enable GradDebugCallback to inspect non-finite gradients before optimizer step."},
    )

    def build_distill_config(self) -> DistillationConfig:
        """Build a :class:`DistillationConfig` from these CLI training arguments."""
        return DistillationConfig(
            teacher=self.teacher_model_name_or_path,
            loss=self.kd_loss_type,
            top_k=self.kd_top_k,
            temperature=self.kd_temperature,
            reverse=self.kd_reverse_kl,
            kd_weight=self.kd_loss_weight,
            ce_weight=self.ce_loss_weight,
            chunk_size=self.logits_chunk_size,
            teacher_device=self.teacher_placement,
        )

    def build_quant_config(self) -> Optional[FakeQuantConfig]:
        """Load :class:`FakeQuantConfig` if ``quant_recipe`` is specified."""
        if not self.quant_recipe:
            return None
        if self.quant_recipe.endswith(".json"):
            cfg = FakeQuantConfig.from_json_file(self.quant_recipe)
        else:
            cfg = FakeQuantConfig.from_yaml(self.quant_recipe)
        if self.quant_backend and self.quant_backend != "auto":
            cfg.backend = self.quant_backend  # type: ignore[assignment]
        return cfg


__all__ = ["QADTrainingArguments"]
