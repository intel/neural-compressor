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
"""Export quantized hf model to compatible formats."""

import tempfile
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def _export_quantized_weight(sub_module: nn.Module, quantization_format: str = None, weight_name: str = "weight"):
    """For the given weight attr of the sub_module, export the quantization info of it.

    The export includes converting weight tensor to correct quantized values and quantized dtype,
    and registering scaling factors.
    """
    if quantization_format is None:
        return

    weight: nn.Parameter = getattr(sub_module, weight_name)
    weight_quantizer = getattr(sub_module, "weight_quantizer")

    qdq_weight, scale = weight_quantizer._fake_quantize(weight)

    # TODO: support more scale dtype when there are other quantization format except mxfp8/mxfp4
    quantized_weight, e8m0_scale = weight_quantizer.weight_pack(qdq_weight, scale)

    sub_module.register_buffer("weight_scale", e8m0_scale)

    if quantization_format == "MXFP8":
        setattr(sub_module, weight_name, nn.Parameter(quantized_weight, requires_grad=False))

    if quantization_format == "MXFP4":
        delattr(sub_module, weight_name)
        # name aligned for vllm emulation
        sub_module.register_buffer("weight_packed", quantized_weight)


def _export_hf_checkpoint(model: nn.Module, scheme: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Exports the torch model to the packed checkpoint with original HF naming.

    The packed checkpoint will be consumed by the TensorRT-LLM unified converter.

    Args:
        model: the torch model.
        dtype: the weights data type to export the unquantized layers or the default model data type if None.

    Returns:
        post_state_dict: Dict containing quantized weights
        quant_config: config information to export hf_quant_cfg.json
    """
    # Create a model layer pool
    # If `model.model` exists use that, otherwise use `model` itself, e.g., Nemotron-H
    root = getattr(model, "model", model)
    # If that has a `.layers`, use it, otherwise fall back to the object itself
    root = getattr(root, "layers", root)
    layer_pool = {f"model.layers.{name}": sub_module for name, sub_module in root.named_modules()}

    from ..algorithms.qat.quant_utils import get_quant_config, get_quantization_format, is_quantlinear

    # compressored config
    quant_config = get_quant_config(scheme=scheme)

    for name, sub_module in layer_pool.items():
        quantization_format = get_quantization_format(sub_module)
        if quantization_format is not None:
            if is_quantlinear(sub_module):
                _export_quantized_weight(sub_module, quantization_format)

    quantized_state_dict = model.state_dict()

    return quantized_state_dict, quant_config


def export_hf2compressored_model(model: nn.Module, export_dir: Path | str = tempfile.gettempdir(), scheme: str = None):
    """Exports the torch model to the packed checkpoint with original HF naming.

    The packed checkpoint will be consumed by the VLLM.
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    try:
        _, quant_config = _export_hf_checkpoint(model, scheme)
        model.save_pretrained(export_dir)
        model.config.quantization_config = quant_config
        model.config.save_pretrained(export_dir)

    except Exception as e:
        warnings.warn("Cannot export model and config, the state can be saved with torch.save for further inspection.")
        raise e
