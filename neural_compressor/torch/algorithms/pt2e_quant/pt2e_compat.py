# Copyright (c) 2026 Intel Corporation
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
"""Compatibility imports for PT2E quantization APIs."""

from importlib import import_module
from typing import Optional

from neural_compressor.torch.utils import TORCH_VERSION_2_11_0, get_torch_version

_PT2E_MODULES = None
_PT2E_IMPORT_ERROR: Optional[ModuleNotFoundError] = None


def _load_pt2e_modules():
    if get_torch_version() >= TORCH_VERSION_2_11_0:
        try:
            pt2e_module = import_module("torchao.quantization.pt2e")
            quantizer_module = import_module("torchao.quantization.pt2e.quantizer.x86_inductor_quantizer")
            xnnpack_module = import_module("torchao.quantization.pt2e.quantizer.xnnpack_quantizer")
            ao_quantization_module = None
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "torch>=2.11 requires `torchao` for PT2E quantization. Please install torchao."
            ) from exc
    else:
        pt2e_module = import_module("torch.ao.quantization.quantize_pt2e")
        quantizer_module = import_module("torch.ao.quantization.quantizer.x86_inductor_quantizer")
        xnnpack_module = import_module("torch.ao.quantization.quantizer.xnnpack_quantizer")
        ao_quantization_module = import_module("torch.ao.quantization")
    return pt2e_module, quantizer_module, xnnpack_module, ao_quantization_module


def _get_pt2e_modules():
    global _PT2E_MODULES
    global _PT2E_IMPORT_ERROR

    if _PT2E_MODULES is None:
        try:
            _PT2E_MODULES = _load_pt2e_modules()
            _PT2E_IMPORT_ERROR = None
        except ModuleNotFoundError as exc:
            _PT2E_IMPORT_ERROR = exc
            raise
    return _PT2E_MODULES


def is_pt2e_available():
    try:
        _get_pt2e_modules()
    except ModuleNotFoundError:
        return False
    return True


def get_pt2e_import_error():
    try:
        _get_pt2e_modules()
    except ModuleNotFoundError as exc:
        return str(exc)
    return None


def __getattr__(name):
    if name in {
        "prepare_pt2e",
        "convert_pt2e",
        "move_exported_model_to_eval",
        "xiq",
        "xpq",
        "X86InductorQuantizer",
        "QuantizationConfig",
    }:
        pt2e_module, xiq_module, xpq_module, ao_quantization_module = _get_pt2e_modules()
        attr_mapping = {
            "prepare_pt2e": pt2e_module.prepare_pt2e,
            "convert_pt2e": pt2e_module.convert_pt2e,
            "move_exported_model_to_eval": (
                pt2e_module.move_exported_model_to_eval
                if get_torch_version() >= TORCH_VERSION_2_11_0
                else ao_quantization_module.move_exported_model_to_eval
            ),
            "xiq": xiq_module,
            "xpq": xpq_module,
            "X86InductorQuantizer": xiq_module.X86InductorQuantizer,
            "QuantizationConfig": xiq_module.QuantizationConfig,
        }
        return attr_mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "is_pt2e_available",
    "get_pt2e_import_error",
    "prepare_pt2e",
    "convert_pt2e",
    "move_exported_model_to_eval",
    "xiq",
    "xpq",
    "X86InductorQuantizer",
    "QuantizationConfig",
]
