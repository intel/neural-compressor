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
from typing import Union

import numpy as np
import onnx
import onnxruntime as ort
from packaging.version import Version

from neural_compressor.onnxrt.algorithms.weight_only.utility import make_matmul_weight_only_node
from neural_compressor.onnxrt.quantization.config import RTNWeightQuantConfig
from neural_compressor.onnxrt.utils.onnx_model import ONNXModel
from neural_compressor.onnxrt.utils.utility import (
    ONNXRT116_VERSION,
    ONNXRT1161_VERSION,
    dtype_mapping,
    simple_progress_bar,
)


def pad_tensor(weight, group_size, k_blocks):
    """Pad tensor rowi so that it can be is divisible by group_size.

    Args:
        weight (array): weight
        group_size (int): how many elements share one scale/zp
        k_blocks (int): the number of block

    Returns:
        weight: paded weight
    """
    if group_size == -1:
        return weight

    org_w_shape = weight.shape
    padded_rows = k_blocks * group_size
    pad_len = padded_rows - org_w_shape[0]

    if pad_len > 0:
        weight = np.pad(weight, ((0, pad_len), (0, 0)), "constant")

    return weight


def quant_tensor(data, num_bits=4, group_size=32, scheme="asym", dtype="int", ratio=1.0):
    """Quantize tensor per group.

    Args:
        data : input weight
        num_bits (int, optional): num_bits. Defaults to 4.
        group_size (int, optional): how many elements share one scale/zp. Defaults to 4.
        scheme (str, optional): quantization scheme. Defaults to "asym".
        dtype (str, optional): data type. Defaults to "int".
        ratio (float, optional): percentile of clip. Defaults to 1.0.

    Returns:
        output: quantized weight
        scale: scale
        zero_point: zero point
    """
    data = np.reshape(data, (-1, group_size))
    if scheme == "asym" or dtype == "uint":
        maxq = 2**num_bits - 1
        minq = 0
    elif scheme == "sym":
        maxq = 2 ** (num_bits - 1) - 1 if num_bits != 1 else 0
        minq = -(2 ** (num_bits - 1)) if num_bits != 1 else -1

    rmin = np.min(data, axis=1, keepdims=True) * ratio
    rmax = np.max(data, axis=1, keepdims=True) * ratio
    if scheme == "sym":
        max_range = np.maximum(np.abs(rmin), np.abs(rmax))
        scale = np.ones(rmax.shape)
        scale[max_range > 0] = np.array(
            [float(i) / (maxq - minq) for i in (max_range[max_range > 0] * 2.0).flatten().tolist()]
        )
        zero_point = (
            np.zeros(scale.shape) if dtype == "int" else np.ones(rmax.shape, dtype="uint8") * (1 << (num_bits - 1))
        )
    else:
        scale = np.ones(rmax.shape)
        scale[rmin != rmax] = np.array(
            [float(i) / (maxq - minq) for i in (rmax - rmin)[rmin != rmax].flatten().tolist()]
        )
        zero_point = (
            ((np.zeros(scale.shape) - rmin) / scale).round()
            if dtype == "int"
            else np.maximum(0, np.minimum(maxq, ((np.zeros(scale.shape) - rmin) / scale).round())).astype("uint8")
        )
    return np.clip((data / scale + zero_point).round(), minq, maxq), scale, zero_point


def qdq_tensor(data, num_bits=4, group_size=32, scheme="asym", dtype="int", ratio=1.0):
    """Quant dequant tensor per group.

    Args:
        data : input weight
        num_bits (int, optional): num_bits. Defaults to 4.
        group_size (int, optional): how many elements share one scale/zp. Defaults to 4.
        scheme (str, optional): quantization scheme. Defaults to "asym".
        dtype (str, optional): data type. Defaults to "int".
        ratio (float, optional): percentile of clip. Defaults to 1.0.

    Returns:
        output: quant-dequant weight
    """
    org_shape = data.shape
    weight, scale, zp = quant_tensor(data, num_bits, group_size, scheme, dtype, ratio)
    return np.reshape(scale * (weight - zp), org_shape)


def rtn_quantize(
    model: Union[onnx.ModelProto, ONNXModel, Path, str],
    weight_config={},
    num_bits=4,
    group_size=32,
    scheme="asym",
    ratios={},
    accuracy_level=0,
    providers=["CPUExecutionProvider"],
):
    """Quant the model with round to nearst method.

    Args:
        model (onnx.ModelProto or ONNXModel or ): onnx model
        weight_config (dict): quantization config
                For example,
                weight_config = {
                    '("/h.0/mlp/fc_out/MatMul", "MatMul")':
                        {
                            'bits': 4,
                            'group_size': 32,
                            'scheme': 'sym',
                            'algorithm': 'RTN'
                        }
                }
        num_bits (int, optional): num_bits. Default is 4.
        group_size (int, optional): how many elements share one scale/zp. Default is 32.
        scheme (str, optional): sym or asym. Defaults to "asym".
        ratios (dict, optional): percentile of clip. Defaults to {}.
        accuracy_level (int): accuracy level. Support 0 (unset), 1(fp32 compute type of jblas kernel),
                              2 (fp16 compute type of jblas kernel), 3 (bf16 compute type of jblas kernel),
                              4 (int8 compute type of jblas kernel)
        providers (list): providers to use

    Returns:
        model: fake quantized ONNXModel
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
        # check node config is RTNWeightQuantConfig
        # check node weight_dtype config is not fp32
        if (
            node.op_type in ["MatMul"]  # check op_type of node is MatMul
            and model.get_initializer(node.input[1]) is not None
            and isinstance(weight_config.get((node.name, node.op_type), {}), RTNWeightQuantConfig)
            and weight_config.get((node.name, node.op_type), {}).weight_dtype.lower() != "fp32"
        ):
            weight_tensor = model.get_initializer(node.input[1])
            weight = onnx.numpy_helper.to_array(weight_tensor, base_dir=base_dir).copy()
            if len(weight.shape) != 2:
                continue

            dtype = weight.dtype

            if (node.name, node.op_type) in weight_config:
                num_bits = weight_config[(node.name, node.op_type)].weight_bits
                group_size = weight_config[(node.name, node.op_type)].weight_group_size
                scheme = "sym" if weight_config[(node.name, node.op_type)].weight_sym else "asym"
                accuracy_level = weight_config[(node.name, node.op_type)].accuracy_level

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
    return model.model


def apply_rtn_on_model(model: onnx.ModelProto, quant_config: RTNWeightQuantConfig) -> onnx.ModelProto:
    providers = quant_config["providers"]

    return rtn_quantize(model, quant_config, providers=providers)
