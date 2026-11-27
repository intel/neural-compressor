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
"""Quantization-Aware Distillation (QAD).

Composes ``neural_compressor.torch.quantization`` (``FakeQuantConfig``, ``prepare_qat``, ``QATTrainer``,
``export_hf_checkpoint``) and ``neural_compressor.torch.distillation`` (``DistillationConfig``, ``KDTrainer``)
into a single HuggingFace-native :class:`QADTrainer`.
"""

from neural_compressor.torch.distillation import DistillationConfig, DistillConfig

from .args import QADTrainingArguments
from .trainer import QADTrainer, prepare_qad_model
from .utils import (
    PEAK_BF16,
    GradDebugCallback,
    StopAfterSteps,
    decoder_layer_class_names,
    device_hardware_name,
    gib,
    memory_report,
    param_counts,
    per_token_flops,
    reset_peak_memory,
    teacher_placement,
    tflops_and_mfu,
)

__all__ = [
    "DistillConfig",
    "DistillationConfig",
    "GradDebugCallback",
    "PEAK_BF16",
    "QADTrainer",
    "QADTrainingArguments",
    "StopAfterSteps",
    "decoder_layer_class_names",
    "device_hardware_name",
    "gib",
    "memory_report",
    "param_counts",
    "per_token_flops",
    "prepare_qad_model",
    "reset_peak_memory",
    "teacher_placement",
    "tflops_and_mfu",
]
