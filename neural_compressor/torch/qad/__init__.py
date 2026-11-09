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

Quantization is inserted by ``quantize(model, FakeQuantConfig, ...)`` from
``neural_compressor.torch.quantization``; this package adds the teacher, the
distillation trainer and export.
"""

from neural_compressor.torch.distill import DistillConfig

from .callbacks import GradDebugCallback, StopAfterSteps
from .config import ExportConfig, ModelConfig, QADConfig, TrainConfig
from .data import PackedDataset, default_collator, make_loader
from .export import export, write_quantization_config
from .modeling import (
    CausalLM,
    GenericAdapter,
    ModelAdapter,
    Qwen35Adapter,
    build_tiny_qwen35,
    get_adapter,
    load_model,
    register_adapter,
)
from .lora import apply_lora, apply_merge_before_quant
from .parallel import decoder_layer_class_names, teacher_placement
from .trainer import QADTrainer

__all__ = [
    "CausalLM",
    "DistillConfig",
    "ExportConfig",
    "GenericAdapter",
    "GradDebugCallback",
    "ModelAdapter",
    "ModelConfig",
    "PackedDataset",
    "QADConfig",
    "QADTrainer",
    "Qwen35Adapter",
    "StopAfterSteps",
    "TrainConfig",
    "apply_lora",
    "apply_merge_before_quant",
    "build_tiny_qwen35",
    "decoder_layer_class_names",
    "default_collator",
    "export",
    "get_adapter",
    "load_model",
    "make_loader",
    "register_adapter",
    "teacher_placement",
    "write_quantization_config",
]
