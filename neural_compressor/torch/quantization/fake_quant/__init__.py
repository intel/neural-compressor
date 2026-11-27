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
"""Fake-quantization foundation layer (formats, QDQ kernels, configs, modules, calibration)."""

from .calibrate import (
    apply_amax_map_to_module_config,
    calibrate,
    calibrate_static_amax,
    canonicalize_module_name,
    canonicalize_quantizer_key,
    get_quantizer_state_dict,
    load_amax_map_from_state,
    load_quantizer_state,
    load_quantizer_state_dict,
    save_quantizer_state,
)
from .config import (
    DEFAULT_IGNORE,
    FP8_DEFAULT_CFG,
    FakeQuantConfig,
    MXFP4_DEFAULT_CFG,
    MXFP4_FP8_HYBRID_CFG,
    MXFP8_DEFAULT_CFG,
    ModuleQuantConfig,
    NVFP4_DEFAULT_CFG,
    NVFP4_FP8_HYBRID_CFG,
    ObserverName,
    QuantSchemeRule,
    QuantizerConfig,
    QuantizerSpec,
    get_default_fake_quant_config,
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
from .modules import QuantLinear
from .plugins import (
    QuantLoraLinear,
    QuantOrder,
    convert_peft_lora_to_quant_lora,
    ensure_peft_torchao_compat,
    is_peft_available,
    merge_lora_into_quant_linear,
    register_peft_quant_modules,
)
from .ops import (
    PackedWeightTensor,
    QuantBackendName,
    StraightThroughFakeQuant,
    dequantize_packed_weight,
    fake_quant_dynamic,
    fake_quant_fp8_static_amax,
    get_quant_backend,
    pack_nibbles,
    pack_weight,
    register_quant_backend,
    set_quant_backend,
    ste_fake_quant,
    unpack_nibbles,
)
from .plan import (
    PlanReport,
    apply_quant_plan,
    build_quant_plan,
    compile_plan_from_info,
    matches_target,
)
from .quantizer import TensorQuantizer, disable_fake_quant
from .registry import (
    QuantFactory,
    is_supported_module,
    quantize_module,
    register_quant_module,
    supported_module_types,
    unregister_quant_module,
)

__all__ = [
    "BF16",
    "DEFAULT_IGNORE",
    "FP8_E4M3",
    "FakeQuantConfig",
    "MXFP4",
    "MXFP8",
    "ModuleQuantConfig",
    "NVFP4",
    "ObserverName",
    "PackedWeightTensor",
    "PlanReport",
    "QuantBackendName",
    "QuantFactory",
    "QuantFormat",
    "QuantLinear",
    "QuantLoraLinear",
    "QuantOrder",
    "QuantSchemeRule",
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
    "dequantize_packed_weight",
    "disable_fake_quant",
    "ensure_peft_torchao_compat",
    "fake_quant_dynamic",
    "fake_quant_fp8_static_amax",
    "get_default_fake_quant_config",
    "get_format",
    "get_quant_backend",
    "get_quantizer_state_dict",
    "is_peft_available",
    "is_supported_module",
    "list_formats",
    "load_amax_map_from_state",
    "load_quantizer_state",
    "load_quantizer_state_dict",
    "matches_target",
    "merge_lora_into_quant_linear",
    "pack_nibbles",
    "pack_weight",
    "quantize_module",
    "register_format",
    "register_peft_quant_modules",
    "register_quant_backend",
    "register_quant_module",
    "save_quantizer_state",
    "set_quant_backend",
    "ste_fake_quant",
    "supported_module_types",
    "unpack_nibbles",
    "unregister_quant_module",
]
