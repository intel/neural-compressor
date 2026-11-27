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
"""Knowledge Distillation package (chunked Top-K KL, online/self/offline teacher, KDTrainer)."""

from .config import DistillConfig, DistillationConfig, DistillationTrainingArguments
from .losses import LogitsKLLoss, TopKLogitsKLLoss, full_vocab_kl, teacher_topk, topk_kl, topk_kl_from_topk
from .offline_topk import (
    PrecomputedTopKCollator,
    PrecomputedTopKDataset,
    extract_teacher_topk_batch,
    save_precomputed_topk_shard,
)
from .seq_packing import pack_sft_sequences

__all__ = [
    "DistillConfig",
    "DistillationConfig",
    "DistillationTrainingArguments",
    "KDTrainer",
    "LogitsKLLoss",
    "PrecomputedTopKCollator",
    "PrecomputedTopKDataset",
    "TopKLogitsKLLoss",
    "extract_teacher_topk_batch",
    "full_vocab_kl",
    "pack_sft_sequences",
    "save_precomputed_topk_shard",
    "teacher_topk",
    "topk_kl",
    "topk_kl_from_topk",
]


def __getattr__(name: str):
    if name == "KDTrainer":
        from .trainer import KDTrainer

        return KDTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
