# Copyright (c) 2024 Intel Corporation
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
"""Utility functions for PT2E quantization."""

from typing import Dict

import torch

from neural_compressor.torch.algorithms.pt2e_quant.pt2e_compat import QuantizationConfig, X86InductorQuantizer, xiq
from neural_compressor.torch.utils import GT_OR_EQUAL_TORCH_VERSION_2_5, TORCH_VERSION_2_11_0, get_torch_version, logger

if get_torch_version() >= TORCH_VERSION_2_11_0:
    from torchao.quantization.pt2e.quantizer import QuantizationSpec
else:
    from torch.ao.quantization.quantizer import QuantizationSpec


def create_quant_spec_from_config(dtype, sym, granularity, algo, is_dynamic=False) -> QuantizationSpec:
    """Create a quantization specification based on the given configuration.

    Args:
        dtype (str): The desired data type for quantization. Valid options are "int8" and "uint8".
        sym (bool): Whether to use symmetric quantization or not.
        granularity (str): The granularity of quantization. Valid options are "per_channel" and "per_tensor".
        algo (str): The algorithm to use for quantization. Valid options are "placeholder", "minmax", and "kl".
        is_dynamic (bool, optional): Whether to use dynamic quantization or not. Defaults to False.

    Returns:
        QuantizationSpec: The created quantization specification.
    """
    dtype_mapping: Dict[str, torch.dtype] = {"int8": torch.int8, "uint8": torch.uint8}
    select_dtype = dtype_mapping[dtype]
    min_max_mapping = {torch.int8: (-128, 127), torch.uint8: (0, 255)}
    qscheme_mapping = {
        "per_channel": {True: torch.per_channel_symmetric, False: torch.per_tensor_affine},
        "per_tensor": {True: torch.per_tensor_symmetric, False: torch.per_tensor_affine},
    }
    observer_mapping = {
        "placeholder": xiq.PlaceholderObserver,
        "minmax": getattr(xiq, "MovingAverageMinMaxObserver"),
        "kl": getattr(xiq, "HistogramObserver"),
        "per_channel_minmax": getattr(xiq, "PerChannelMinMaxObserver"),
    }
    # Dynamic PT2E quantization on torchao expects PlaceholderObserver from the PT2E quantizer module.
    if is_dynamic:
        algo = "placeholder"
    if f"{granularity}_{algo}" in observer_mapping:
        observer_or_fake_quant_ctr = observer_mapping[f"{granularity}_{algo}"]
    else:
        observer_or_fake_quant_ctr = observer_mapping[algo]
    # qscheme
    qscheme = qscheme_mapping[granularity][sym]
    quantization_spec = QuantizationSpec(
        dtype=select_dtype,
        quant_min=min_max_mapping[select_dtype][0],
        quant_max=min_max_mapping[select_dtype][1],
        observer_or_fake_quant_ctr=observer_or_fake_quant_ctr,
        ch_axis=0 if granularity == "per_channel" else None,
        qscheme=qscheme,
        is_dynamic=is_dynamic,
    )
    return quantization_spec


def _map_inc_config_to_torch_quant_config(inc_config, is_dynamic=False) -> QuantizationConfig:
    NOT_QUANT_DTYPES = ["fp32", "fp16", "bf16"]
    if inc_config.act_dtype in NOT_QUANT_DTYPES and inc_config.w_dtype in NOT_QUANT_DTYPES:  # pragma: no cover
        logger.debug("Got non-quantizable data types, skipping quantization.")
        return None
    input_act_quant_spec = create_quant_spec_from_config(
        inc_config.act_dtype, inc_config.act_sym, inc_config.act_granularity, inc_config.act_algo, is_dynamic=is_dynamic
    )
    output_act_quant_spec = create_quant_spec_from_config(
        inc_config.act_dtype, inc_config.act_sym, inc_config.act_granularity, inc_config.act_algo, is_dynamic=is_dynamic
    )
    weight_quant_spec = create_quant_spec_from_config(
        inc_config.w_dtype, inc_config.w_sym, inc_config.w_granularity, inc_config.w_algo
    )
    default_quant_config = xiq.get_default_x86_inductor_quantization_config(is_dynamic=is_dynamic)
    quant_config = QuantizationConfig(
        input_activation=input_act_quant_spec,
        output_activation=output_act_quant_spec,
        weight=weight_quant_spec,
        bias=default_quant_config.bias,
        is_qat=False,
    )
    return quant_config


def create_default_xiq_quantizer_config(is_dynamic=False) -> QuantizationConfig:
    """Create the default PT2E config using INC defaults for both input and output activations."""
    default_quant_config = xiq.get_default_x86_inductor_quantization_config(is_dynamic=is_dynamic)
    act_quant_spec = create_quant_spec_from_config(
        dtype="uint8",
        sym=False,
        granularity="per_tensor",
        algo="minmax",
        is_dynamic=is_dynamic,
    )
    weight_quant_spec = create_quant_spec_from_config(
        dtype="int8",
        sym=True,
        granularity="per_channel",
        algo="minmax",
    )
    return QuantizationConfig(
        input_activation=act_quant_spec,
        output_activation=act_quant_spec,
        weight=weight_quant_spec,
        bias=default_quant_config.bias,
        is_qat=False,
    )


def create_xiq_quantizer_from_pt2e_config(config, is_dynamic=False) -> X86InductorQuantizer:
    """Creates an instance of X86InductorQuantizer based on the given configuration.

    Args:
        config: The configuration object containing the quantization settings.
        is_dynamic: A boolean indicating whether dynamic quantization is enabled.

    Returns:
        An instance of X86InductorQuantizer initialized with the provided configuration.
    """
    quantizer = xiq.X86InductorQuantizer()
    # set global
    global_config = _map_inc_config_to_torch_quant_config(config, is_dynamic)
    quantizer.set_global(global_config)
    # need torch >= 2.5
    if GT_OR_EQUAL_TORCH_VERSION_2_5:  # pragma: no cover
        op_type_config_dict, op_name_config_dict = config._get_op_name_op_type_config()
        if op_type_config_dict:
            for op_type, config in op_type_config_dict.items():
                _nn_module_type = getattr(torch.nn, op_type, None)
                if _nn_module_type:
                    quantizer.set_module_type_qconfig(
                        _nn_module_type, _map_inc_config_to_torch_quant_config(config, is_dynamic)
                    )
                _nn_func_type = getattr(torch.nn.functional, op_type, None)
                if _nn_func_type:
                    quantizer.set_function_type_qconfig(
                        _nn_func_type, _map_inc_config_to_torch_quant_config(config, is_dynamic)
                    )
        if op_name_config_dict:
            for op_name, config in op_name_config_dict.items():
                quantizer.set_module_name_qconfig(op_name, _map_inc_config_to_torch_quant_config(config, is_dynamic))
    return quantizer
