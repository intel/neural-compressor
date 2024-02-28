#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 MIT HAN Lab
# This source code is licensed under the MIT license
#
# Copyright (c) 2023 Intel Corporation
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


import os
from pathlib import Path
from typing import List, Union

import numpy as np
import onnx
import onnxruntime as ort
from packaging.version import Version

from neural_compressor.onnxrt.algorithms.weight_only.utility import (
    make_matmul_weight_only_node,
    pad_tensor,
    qdq_tensor,
    quant_tensor,
)
from neural_compressor.onnxrt.quantization.config import RTNConfig
from neural_compressor.onnxrt.utils.onnx_model import ONNXModel
from neural_compressor.onnxrt.utils.utility import (
    ONNXRT116_VERSION,
    ONNXRT1161_VERSION,
    dtype_mapping,
    simple_progress_bar,
)

__all__ = ["apply_rtn_on_model", "rtn_quantize"]


def rtn_quantize(
    model: Union[onnx.ModelProto, ONNXModel, Path, str],
    weight_config: dict = {},
    num_bits: int = 4,
    group_size: int = 32,
    scheme: str = "asym",
    ratios: dict = {},
    accuracy_level: int = 0,
    providers: List[str] = ["CPUExecutionProvider"],
    return_modelproto: bool = True,
):
    """Quantize the model with round to nearst method.

    Args:
        model (Union[onnx.ModelProto, ONNXModel, Path, str]): onnx model
        weight_config (dict, optional): quantization config
            For example,
            weight_config = {
                '(fc2, "MatMul")':
                    {
                        'weight_dtype': 'int',
                        'weight_bits': 4,
                        'weight_group_size': 32,
                        'weight_sym': True,
                        'accuracy_level': 0
                    }
            }. Defaults to {}.
        num_bits (int, optional): number of bits used to represent weights. Defaults to 4.
        group_size (int, optional): size of weight groups. Defaults to 32.
        scheme (str, optional): indicates whether weights are symmetric. Defaults to "asym".
        ratios (dict, optional): percentile of clip. Defaults to {}.
        accuracy_level (int, optional):
            accuracy level. Support 0 (unset), 1(fp32 compute type of jblas kernel),
            2 (fp16 compute type of jblas kernel), 3 (bf16 compute type of jblas kernel),
            4 (int8 compute type of jblas kernel). Defaults to 0.
        providers (list, optional): providers to use. Defaults to ["CPUExecutionProvider"].
        return_modelproto (bool, optionmal): whether to return onnx.Modelproto. set False for layer-wise quant.
            Default to True
    Returns:
        onnx.ModelProto: quantized onnx model.
    """
    if not isinstance(model, ONNXModel):
        model = ONNXModel(model)
    base_dir = os.path.dirname(model.model_path) if model.model_path is not None else ""
    new_nodes = []
    remove_nodes = []
    total_num = len([i for i in model.nodes() if i.op_type in ["MatMul"]])
    curr_id = 0
    for node in model.nodes():
        if node.op_type in ["MatMul"]:
            curr_id += 1
            simple_progress_bar(total_num, curr_id)

        # check op_type of node is MatMul
        # check dim 1 of input is weight tensor
        # check weight_type is not "fp32"
        if (
            node.op_type in ["MatMul"]  # check op_type of node is MatMul
            and model.get_initializer(node.input[1]) is not None
            and weight_config.get((node.name, node.op_type), {}).get("weight_dtype", "fp32") != "fp32"
        ):
            weight_tensor = model.get_initializer(node.input[1])
            weight = onnx.numpy_helper.to_array(weight_tensor, base_dir=base_dir).copy()
            if len(weight.shape) != 2:
                continue

            dtype = weight.dtype
            if (node.name, node.op_type) in weight_config:
                num_bits = weight_config[(node.name, node.op_type)].get("weight_bits", 4)
                group_size = weight_config[(node.name, node.op_type)].get("weight_group_size", 32)
                scheme = "sym" if weight_config[(node.name, node.op_type)].get("weight_sym", True) else "asym"
                accuracy_level = weight_config[(node.name, node.op_type)].get("accuracy_level", 0)

            org_w_shape = weight.shape  # ic, oc
            group_size = group_size if group_size != -1 else org_w_shape[0]

            k_blocks = (org_w_shape[0] - 1) // group_size + 1
            init_share_num = model.get_initializer_share_num(node.input[1])

            weight = pad_tensor(weight, group_size, k_blocks)

            satisfy_MatMulNBits_condition = Version(ort.__version__) > ONNXRT1161_VERSION and num_bits == 4
            satisfy_MatMulFpQ4_condition = (
                Version(ort.__version__) >= ONNXRT116_VERSION and num_bits == 4 and group_size == 32
            )
            if ("CUDAExecutionProvider" in providers and satisfy_MatMulNBits_condition) or (
                "CUDAExecutionProvider" not in providers
                and (satisfy_MatMulFpQ4_condition or satisfy_MatMulNBits_condition)
            ):  # pragma: no cover
                # MatMulFpQ4 support 4 bits and 32 group_size with ort 1.16.0 and 1.16.1 versions, supported by CPU EP
                # MatMulNBits supports 4 bits and 2^n group_size with ort > 1.16.1, supported by CPU EP AND CUDA EP
                q_weight, scale, zp = quant_tensor(
                    weight.T, num_bits, group_size, scheme, "uint", ratios.get(node.input[1], 1)
                )
                q_matmul_node, new_inits = make_matmul_weight_only_node(
                    node=node,
                    weight_shape=org_w_shape,
                    num_bits=num_bits,
                    group_size=group_size,
                    k_blocks=k_blocks,
                    q_weight=q_weight.astype("uint8"),
                    scale=scale.astype(dtype),
                    zero_point=zp if scheme == "asym" else None,
                    accuracy_level=accuracy_level,
                )

                model.add_initializers(new_inits)
                remove_nodes.append(node)
                new_nodes.append(q_matmul_node)
            else:
                q_weight = qdq_tensor(weight.T, num_bits, group_size, scheme, "int", ratios.get(node.input[1], 1))
                q_weight = np.reshape(q_weight, (org_w_shape[1], -1))
                q_weight = np.transpose(q_weight)
                q_weight = q_weight[: org_w_shape[0], :].astype(dtype)
                q_weight_tensor = onnx.helper.make_tensor(
                    name=node.input[1] + "_Q{}G{}".format(str(num_bits), str(group_size)),
                    data_type=dtype_mapping[str(dtype)],
                    dims=weight.shape,
                    vals=q_weight.tobytes(),
                    raw=True,
                )
                model.add_initializer(q_weight_tensor)
                node.input[1] = q_weight_tensor.name
            if init_share_num == 1:
                model.remove_initializer(weight_tensor)

    model.add_nodes(new_nodes)
    model.remove_nodes(remove_nodes)
    model.topological_sort()

    # reload external data to prevent external data file path errors
    if model.is_large_model:
        from onnx.external_data_helper import load_external_data_for_model

        load_external_data_for_model(model.model, os.path.split(model.model_path)[0])

    if return_modelproto:
        return model.model
    else:
        return model


def apply_rtn_on_model(model: Union[onnx.ModelProto, ONNXModel, Path, str], quant_config: dict) -> onnx.ModelProto:
    """Apply RTN on onnx model.

    Args:
        model (Union[onnx.ModelProto, ONNXModel, Path, str]): onnx model.
        quant_config (dict): quantization config.

    Returns:
        onnx.ModelProto: quantized onnx model.
    """
    # check whether to do layer_wise quant
    layer_wise = quant_config.pop("layer_wise_quant", False)

    # set other model params
    quant_kwargs = {}
    quant_kwargs = {key: quant_config.pop(key) for key in RTNConfig.model_params_list if key in quant_config}

    # change op config to dict type
    for op_name_type, op_config in quant_config.items():
        if isinstance(op_config, RTNConfig):
            quant_config[op_name_type] = op_config.to_dict()

    if layer_wise:
        from neural_compressor.onnxrt.algorithms import layer_wise_quant

        quantized_model = layer_wise_quant(model, quant_func=rtn_quantize, weight_config=quant_config, **quant_kwargs)
    else:
        quantized_model = rtn_quantize(model, weight_config=quant_config, **quant_kwargs)

    if isinstance(quantized_model, ONNXModel):
        quantized_model = quantized_model.model
    return quantized_model
