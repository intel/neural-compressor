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
"""Knowledge distillation: backward-compatible re-export of ``neural_compressor.torch.distillation``."""

from neural_compressor.torch.distillation import (
    DistillConfig,
    DistillationConfig,
    PrecomputedTopKCollator,
    PrecomputedTopKDataset,
    TopKLogitsKLLoss,
    full_vocab_kl,
    pack_sft_sequences,
    teacher_topk,
    topk_kl,
    topk_kl_from_topk,
)

__all__ = [
    "DistillConfig",
    "DistillationConfig",
    "KDTrainer",
    "PrecomputedTopKCollator",
    "PrecomputedTopKDataset",
    "TopKLogitsKLLoss",
    "full_vocab_kl",
    "pack_sft_sequences",
    "teacher_topk",
    "topk_kl",
    "topk_kl_from_topk",
]


def __getattr__(name: str):
    if name == "KDTrainer":
        from neural_compressor.torch.distillation.trainer import KDTrainer

        return KDTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
