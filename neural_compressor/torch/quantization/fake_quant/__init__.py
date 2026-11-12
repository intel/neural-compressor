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
"""Fake quantization: mechanism shared by QAT and QAD."""

from .calibrator import AmaxMap, apply_amax, calibrate, load_amax, modelopt_megatron_translate
from .config import FakeQuantConfig
from .formats import BF16, FP8_E4M3, MXFP4, NVFP4, QuantFormat, get_format, list_formats, register_format
from .modules import QuantLinear
from .options import ModuleQuantConfig, QuantizerConfig
from .plan import (
    DEFAULT_EXCLUDE,
    PlanReport,
    QuantRule,
    apply_plan,
    compile_plan,
    compile_plan_from_info,
    print_plan_summary,
    summarize_plan,
)
from .quantizer import TensorQuantizer
from .registry import QuantModuleRegistry

__all__ = [
    "AmaxMap",
    "BF16",
    "DEFAULT_EXCLUDE",
    "FakeQuantConfig",
    "FP8_E4M3",
    "ModuleQuantConfig",
    "MXFP4",
    "NVFP4",
    "PlanReport",
    "QuantFormat",
    "QuantLinear",
    "QuantModuleRegistry",
    "QuantRule",
    "QuantizerConfig",
    "TensorQuantizer",
    "apply_amax",
    "apply_plan",
    "calibrate",
    "compile_plan",
    "compile_plan_from_info",
    "get_format",
    "list_formats",
    "load_amax",
    "modelopt_megatron_translate",
    "print_plan_summary",
    "register_format",
    "summarize_plan",
]
