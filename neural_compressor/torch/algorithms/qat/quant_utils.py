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
"""Utils for quantization."""

import types
from typing import Any

import torch
import torch.nn as nn

from .quant_linear import QuantLinear
from .tensor_quantizer import TensorQuantizer


def convert(module: nn.Module, quant_cfg=None, quant_module=None):
    """Convert the model to a quantized one with quant config."""

    # update class
    original_cls = type(module)
    module.__class__ = quant_module
    module.forward = types.MethodType(quant_module.forward, module)

    # setup quantizers
    module._setup(quant_cfg)

    return module


def replace_with_quant_linear(model, quant_cfg=None):
    """Recursively replace the module with quantized module."""

    # TODO: support more modules, like kv.
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            if "lm_head" in name:
                continue
            # REPLACE on the parent (model), not on child
            quantized = convert(child, quant_cfg, QuantLinear)
            setattr(model, name, quantized)

        # now recurse into whichever module is now at `model.name`
        replace_with_quant_linear(getattr(model, name), quant_cfg=quant_cfg)

    return model


def get_quant_config_with_scheme(scheme: str):
    """Get quantization config."""

    try:
        # use scheme definitions from AutoRound since we utilize the quantization functions now
        from auto_round.schemes import preset_name_to_scheme

        quant_cfg = preset_name_to_scheme(scheme)
        return quant_cfg
    except ImportError:
        return None


def convert_model_with_mapping(model, mapping=None):
    """Process mapping to quant config."""
    # key is torch module, TODO: support more key format, like layer name.
    for key in mapping:
        # TODO: support more torch modules
        if key == nn.Linear:
            quant_cfg = get_quant_config_with_scheme(mapping[key])
            if quant_cfg is None:
                continue
            replace_with_quant_linear(model, quant_cfg)

    replaced_modules = sum(isinstance(m, TensorQuantizer) for _, m in model.named_modules())
    print(f"Inserted {replaced_modules} quantizers")


def get_quant_config(scheme: str) -> dict[str, Any]:
    """Generate quantization config for a torch model.

    Args:
        model: The PyTorch model to analyze

    Returns:
        Dictionary containing the quantization configuration
    """

    # TODO: support more quant config
    try:
        from auto_round.export.export_to_llmcompressor.config import initialize_quantization

        quantization_config = initialize_quantization(scheme=scheme)
        quantization_config = quantization_config.to_dict()
        quantization_config["provider"] = "auto-round"
        quantization_config["config_groups"]["group_0"]["weights"]["is_mx"] = True
        quantization_config["config_groups"]["group_0"]["input_activations"]["is_mx"] = True
        quantization_config["format"] = "float-quantized"

    except ImportError:
        quantization_config = None

    return quantization_config


def get_quantization_format(module) -> str | None:
    """Gets the quantization string.

    Gets the quantization string by iterating through the module and its children.
    The first non-None quantization string is returned.
    """

    def _get_quantization_from_layer(layer):
        weight_quantizer = getattr(layer, "weight_quantizer", None)
        input_quantizer = getattr(layer, "input_quantizer", None)

        if weight_quantizer is None or weight_quantizer._disabled:
            return None

        # TODO: support more quant format
        if weight_quantizer.num_bits == 8 and weight_quantizer.data_type == "mx_fp8":
            return "MXFP8"

        if weight_quantizer.num_bits == 4 and weight_quantizer.data_type == "mx_fp4":
            return "MXFP4"

        # Raise error for unsupported num_bits
        raise NotImplementedError(f"Unsupported quantizer with num_bits: {weight_quantizer.num_bits}")

    quantization = _get_quantization_from_layer(module)
    if quantization is not None:
        return quantization

    for _, layer in module.named_children():
        format = get_quantization_format(layer)
        if format is not None:
            return format

    return None


def is_quantlinear(module: nn.Module) -> bool:
    """Returns whether the module is a quantized linear layer."""
    return "QuantLinear" in type(module).__name__
