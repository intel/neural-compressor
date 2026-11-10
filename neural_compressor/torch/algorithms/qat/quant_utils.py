#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Export helpers for fake-quant/QAT models.

Only the HF/compressed-tensors export helpers remain here. Model conversion
moved to :mod:`neural_compressor.torch.quantization.fake_quant` (see
``FakeQuantConfig`` and ``prepare_qat``), so this module no longer mutates
module classes or builds quantized models.
"""

from __future__ import annotations

from typing import Any


def get_quant_config(scheme: str) -> dict[str, Any] | None:
    """Build a compressed-tensors quantization config for ``scheme``.

    Kept for the legacy export path; requires ``auto-round`` when called.
    """
    try:
        from auto_round.export.export_to_llmcompressor.config import initialize_quantization

        quantization_config = initialize_quantization(scheme=scheme)
        quantization_config = quantization_config.to_dict()
        quantization_config["provider"] = "auto-round"
        quantization_config["config_groups"]["group_0"]["weights"]["is_mx"] = True
        quantization_config["config_groups"]["group_0"]["input_activations"]["is_mx"] = True
        quantization_config["format"] = "float-quantized"
        return quantization_config
    except ImportError:
        return None


def get_quantization_format(module) -> str | None:
    """Return the first ``"MXFP8"``/``"MXFP4"`` format in a module tree."""
    weight_quantizer = getattr(module, "weight_quantizer", None)
    if weight_quantizer is not None and not getattr(weight_quantizer, "_disabled", True):
        num_bits = getattr(weight_quantizer, "num_bits", None)
        data_type = getattr(weight_quantizer, "data_type", None)
        if num_bits == 8 and data_type == "mx_fp8":
            return "MXFP8"
        if num_bits == 4 and data_type == "mx_fp4":
            return "MXFP4"
        raise NotImplementedError(f"Unsupported quantizer with num_bits={num_bits}, data_type={data_type}")
    for _, child in module.named_children():
        quantization = get_quantization_format(child)
        if quantization is not None:
            return quantization
    return None


def is_quantlinear(module) -> bool:
    """Return whether the module is a quantized linear layer."""
    return "QuantLinear" in type(module).__name__


__all__ = ["get_quant_config", "get_quantization_format", "is_quantlinear"]
