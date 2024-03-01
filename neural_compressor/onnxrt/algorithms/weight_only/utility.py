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

import struct
import sys

import numpy as np
import onnx
import onnxruntime as ort
from packaging.version import Version

from neural_compressor.onnxrt.utils.utility import ONNXRT1161_VERSION, dtype_mapping

__all__ = [
    "make_matmul_weight_only_node",
    "prepare_inputs",
    "pad_tensor",
    "quant_tensor",
    "qdq_tensor",
]


def _get_blob_size(group_size, has_zp):  # pragma: no cover
    """Get blob_size.

    Args:
        group_size (int): how many elements share one scale/zp
        has_zp (bool): whether zero_point is None
    """
    if Version(ort.__version__) > ONNXRT1161_VERSION:
        blob_size = group_size // 2
    elif has_zp:
        blob_size = group_size // 2 + 4 + 1
    else:
        blob_size = group_size // 2 + 4
    return blob_size


def make_matmul_weight_only_node(
    node: onnx.NodeProto,
    weight_shape: tuple,
    num_bits: int,
    group_size: int,
    k_blocks: int,
    q_weight: np.array,
    scale: np.array,
    zero_point: np.array,
    accuracy_level: int = 0,
):
    """Build MatMulFpQ4/MatMulNBits node.

    Args:
        node (onnx.NodeProto): original matmul node
        weight_shape (tuple): original weight shape
        num_bits (int): number of bits used to represent weights.
        group_size (int): how many elements share one scale/zp
        k_blocks (int): block number
        q_weight (np.array): quantized weight
        scale (np.array): scale
        zero_point (np.array): zero point
        accuracy_level (int, optional): accuracy level.
            Support 0 (unset), 1(fp32 compute type of jblas kernel),
            2 (fp16 compute type of jblas kernel), 3 (bf16 compute type of jblas kernel),
            4 (int8 compute type of jblas kernel) Defaults to 0.

    Returns:
        matmul_weight_only_node: MatMulFpQ4 or MatMulNBits node
        new_inits: initializers of the new node
    """
    blob_size = _get_blob_size(group_size, zero_point is not None)
    packed = np.zeros((q_weight.shape[0], blob_size), dtype="uint8")
    q_weight_name = node.input[1] + "_Q{}G{}".format(str(num_bits), str(group_size))
    input_names = [node.input[0], q_weight_name]
    new_inits = []
    kwargs = {}

    if Version(ort.__version__) > ONNXRT1161_VERSION:
        op_type = "MatMulNBits"

        # pack quantized weight
        for i in range(q_weight.shape[0]):
            for k in range(0, group_size, 2):
                packed[i][k // 2] = q_weight[i][k] | q_weight[i][k + 1] << 4
        packed = np.reshape(packed, (-1, k_blocks, blob_size))

        # build scale tensor
        scale = np.reshape(scale, (-1, k_blocks))
        scale_tensor = onnx.helper.make_tensor(
            name=node.input[1] + "_scale",
            data_type=dtype_mapping[str(scale.dtype)],
            dims=scale.shape,
            vals=scale.tobytes(),
            raw=True,
        )
        input_names.append(scale_tensor.name)
        new_inits.append(scale_tensor)

        # build zero_point tensor
        if zero_point is not None:
            if num_bits > 4:
                packed_zp = np.reshape(zero_point, (1, -1)).astype("uint8")
            else:
                packed_zp = np.full((zero_point.shape[0] + 1) // 2, 136, dtype="uint8")
                for i in range(zero_point.shape[0] // k_blocks):
                    for j in range(k_blocks):
                        idx = i * k_blocks + j
                        zp = zero_point[idx]
                        packed_zp[idx // 2] = (
                            ((packed_zp[idx // 2] & 0x0F) | (zp << 4))
                            if (idx & 1)
                            else ((packed_zp[idx // 2] & 0xF0) | zp)
                        )

            zp_tensor = onnx.helper.make_tensor(
                name=node.input[1] + "_zp", data_type=2, dims=packed_zp.shape, vals=packed_zp.tobytes(), raw=True
            )
            input_names.append(zp_tensor.name)
            new_inits.append(zp_tensor)

        # set kwargs
        kwargs["K"] = weight_shape[0]
        kwargs["N"] = weight_shape[1]
        kwargs["bits"] = num_bits
        kwargs["block_size"] = group_size
        if accuracy_level > 0:
            # require onnxruntime > 1.16.3
            kwargs["accuracy_level"] = accuracy_level

    else:
        offset = 5 if zero_point is not None else 4
        op_type = "MatMulFpQ4"

        # pack quantized weight
        for i in range(q_weight.shape[0]):
            bf = struct.pack("f", scale[i])
            packed[i][0] = bf[0]
            packed[i][1] = bf[1]
            packed[i][2] = bf[2]
            packed[i][3] = bf[3]

            if zero_point is not None:
                packed[i][4] = zero_point[i]

            packed[i][offset:] = np.bitwise_or(
                q_weight[i][: group_size // 2], np.left_shift(q_weight[i][group_size // 2 :], num_bits)
            )
        packed = packed.reshape(-1)

        # build shape tensor
        shape_tensor = onnx.helper.make_tensor(
            name=node.input[1] + "_shape", data_type=7, dims=(2,), vals=np.array(weight_shape, dtype="int64")
        )
        new_inits.append(shape_tensor)
        input_names.append(shape_tensor.name)

        # set kwargs
        kwargs["blk_quant_type"] = 1 if zero_point is not None else 0

    q_weight_tensor = onnx.helper.make_tensor(
        name=q_weight_name,
        data_type=2,
        dims=packed.shape,
        vals=packed.tobytes(),
        raw=True,
    )
    new_inits.append(q_weight_tensor)

    matmul_weight_only_node = onnx.helper.make_node(
        op_type,
        inputs=input_names,
        outputs=node.output,
        name=node.name + "_Q" + str(num_bits) if node.name else "_Q" + str(num_bits),
        domain="com.microsoft",
        **kwargs,
    )
    return matmul_weight_only_node, new_inits


def prepare_inputs(model, data_reader, providers):
    """Prepare inputs for weight only quantization.

    Args:
        model (ModelProto or ONNXModel): onnx model.
        data_reader (CalibrationDataReader): a calibration data reader.
        providers (list): providers to use.

    Returns:
        inputs: prepared inputs.
        so: session options
    """
    from importlib.util import find_spec

    so = ort.SessionOptions()
    if sys.version_info < (3, 11) and find_spec("onnxruntime_extensions"):  # pragma: no cover
        from onnxruntime_extensions import get_library_path

        so.register_custom_ops_library(get_library_path())
    if model.is_large_model:
        onnx.save_model(
            model.model,
            model.model_path + "_augment.onnx",
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            convert_attribute=False,
        )

    inputs_list = []
    while True:
        inputs = data_reader.get_next()
        if not inputs:
            break
        inputs_list.append(inputs)
    return inputs_list, so


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


def quant_tensor(
    data: np.array,
    num_bits: int = 4,
    group_size: int = 32,
    scheme: str = "asym",
    dtype: str = "int",
    ratio: float = 1.0,
):
    """Quantize tensor per group.

    Args:
        data (np.array): input weight
        num_bits (int, optional): number of bits used to represent weights. Defaults to 4.
        group_size (int, optional): how many elements share one scale/zp. Defaults to 4.
        scheme (str, optional): _quantization scheme. Defaults to "asym".
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


def qdq_tensor(
    data: np.array,
    num_bits: int = 4,
    group_size: int = 32,
    scheme: str = "asym",
    dtype: str = "int",
    ratio: float = 1.0,
):
    """Quant dequant tensor per group.

    Args:
        data (np.array): input weight
        num_bits (int, optional): number of bits used to represent weights. Defaults to 4.
        group_size (int, optional):  how many elements share one scale/zp. Defaults to 32.
        scheme (str, optional): quantization scheme. Defaults to "asym".
        dtype (str, optional): data type. Defaults to "int".
        ratio (float, optional): percentile of clip. Defaults to 1.0.

    Returns:
        output: quant-dequant weight
    """
    org_shape = data.shape
    weight, scale, zp = quant_tensor(data, num_bits, group_size, scheme, dtype, ratio)
    return np.reshape(scale * (weight - zp), org_shape)
