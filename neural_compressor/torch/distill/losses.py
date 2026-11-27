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
"""Backward-compatible re-export of ``neural_compressor.torch.distillation.losses``."""

from neural_compressor.torch.distillation.losses import (
    TopKLogitsKLLoss,
    full_vocab_kl,
    teacher_topk,
    topk_kl,
    topk_kl_from_topk,
)

__all__ = ["TopKLogitsKLLoss", "full_vocab_kl", "teacher_topk", "topk_kl", "topk_kl_from_topk"]
