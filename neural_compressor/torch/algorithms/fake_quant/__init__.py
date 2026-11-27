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
"""Fake-quantization algorithm implementation (formats, functional QDQ, modules, conversion, calibration)."""

from __future__ import annotations

from .calib import calibrate, calibrate_static_amax
from .config import (
    DEFAULT_IGNORE,
    FP8_DEFAULT_CFG,
    MXFP4_DEFAULT_CFG,
    MXFP4_FP8_HYBRID_CFG,
    MXFP8_DEFAULT_CFG,
    NVFP4_DEFAULT_CFG,
    NVFP4_FP8_HYBRID_CFG,
    FakeQuantConfig,
    ModuleQuantConfig,
    ObserverName,
    QuantizerConfig,
    QuantizerSpec,
    QuantSchemeRule,
    get_default_fake_quant_config,
)
from .conversion import (
    PlanReport,
    QuantConversionReport,
    QuantFactory,
    apply_quant_plan,
    build_quant_plan,
    compile_plan_from_info,
    convert_to_quantized_model,
    fake_quant_entry,
    is_supported_module,
    match_quant_config,
    match_quant_config_from_info,
    matches_target,
    quantize_module,
    register_quant_module,
    replace_quant_modules,
    supported_module_types,
    unregister_quant_module,
)
from .formats import (
    BF16,
    FP8_E4M3,
    MXFP4,
    MXFP8,
    NVFP4,
    QuantFormat,
    get_format,
    list_formats,
    register_format,
)
from .functional import (
    QuantBackendName,
    StraightThroughFakeQuant,
    fake_quant_dynamic,
    fake_quant_fp8_static_amax,
    get_quant_backend,
    register_quant_backend,
    set_quant_backend,
    ste_fake_quant,
)
from .pack import (
    PackedWeightTensor,
    dequantize_packed_weight,
    pack_nibbles,
    pack_weight,
    unpack_nibbles,
)
from .plugins import (
    QuantLoraLinear,
    QuantOrder,
    convert_peft_lora_to_quant_lora,
    ensure_peft_torchao_compat,
    is_peft_available,
    merge_lora_into_quant_linear,
    register_peft_quant_modules,
)
from .quant_linear import QuantLinear
from .save_load import (
    apply_amax_map_to_module_config,
    canonicalize_module_name,
    canonicalize_quantizer_key,
    get_quantizer_state_dict,
    load_amax_map_from_state,
    load_quantizer_state,
    load_quantizer_state_dict,
    save_quantizer_state,
)
from .tensor_quantizer import TensorQuantizer, disable_fake_quant

_TRAINER_EXPORTS = (
    "QATTrainer",
    "QuantizationTrainingArguments",
    "ensure_peft_gradient_checkpointing_delegation",
    "is_peft_model",
    "save_fsdp2_full_model",
    "save_fsdp2_peft_adapter",
    "unwrap_model",
)


def __getattr__(name: str):
    if name in _TRAINER_EXPORTS:
        from . import trainer

        return getattr(trainer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BF16",
    "DEFAULT_IGNORE",
    "FP8_DEFAULT_CFG",
    "FP8_E4M3",
    "FakeQuantConfig",
    "MXFP4",
    "MXFP4_DEFAULT_CFG",
    "MXFP4_FP8_HYBRID_CFG",
    "MXFP8",
    "MXFP8_DEFAULT_CFG",
    "ModuleQuantConfig",
    "NVFP4",
    "NVFP4_DEFAULT_CFG",
    "NVFP4_FP8_HYBRID_CFG",
    "ObserverName",
    "PackedWeightTensor",
    "PlanReport",
    "QATTrainer",
    "QuantBackendName",
    "QuantConversionReport",
    "QuantFactory",
    "QuantFormat",
    "QuantLinear",
    "QuantLoraLinear",
    "QuantOrder",
    "QuantSchemeRule",
    "QuantizationTrainingArguments",
    "QuantizerConfig",
    "QuantizerSpec",
    "StraightThroughFakeQuant",
    "TensorQuantizer",
    "apply_amax_map_to_module_config",
    "apply_quant_plan",
    "build_quant_plan",
    "calibrate",
    "calibrate_static_amax",
    "canonicalize_module_name",
    "canonicalize_quantizer_key",
    "compile_plan_from_info",
    "convert_peft_lora_to_quant_lora",
    "convert_to_quantized_model",
    "dequantize_packed_weight",
    "disable_fake_quant",
    "ensure_peft_gradient_checkpointing_delegation",
    "ensure_peft_torchao_compat",
    "fake_quant_dynamic",
    "fake_quant_entry",
    "fake_quant_fp8_static_amax",
    "get_default_fake_quant_config",
    "get_format",
    "get_quant_backend",
    "get_quantizer_state_dict",
    "is_peft_available",
    "is_peft_model",
    "is_supported_module",
    "list_formats",
    "load_amax_map_from_state",
    "load_quantizer_state",
    "load_quantizer_state_dict",
    "match_quant_config",
    "match_quant_config_from_info",
    "matches_target",
    "merge_lora_into_quant_linear",
    "pack_nibbles",
    "pack_weight",
    "quantize_module",
    "register_format",
    "register_peft_quant_modules",
    "register_quant_backend",
    "register_quant_module",
    "replace_quant_modules",
    "save_fsdp2_full_model",
    "save_fsdp2_peft_adapter",
    "save_quantizer_state",
    "set_quant_backend",
    "ste_fake_quant",
    "supported_module_types",
    "unpack_nibbles",
    "unregister_quant_module",
    "unwrap_model",
]
