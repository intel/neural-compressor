#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 MIT HAN Lab
# This source code is licensed under the MIT license
#
# Copyright (c) 2024 Intel Corporation
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

import numpy as np
import onnx
import onnxruntime as ort
from packaging.version import Version

from neural_compressor.onnxrt.utils.utility import ONNXRT1161_VERSION, dtype_mapping


def get_blob_size(group_size, has_zp):  # pragma: no cover
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
    node,
    weight_shape,
    num_bits,
    group_size,
    k_blocks,
    q_weight,
    scale,
    zero_point,
    accuracy_level=0,
):  # pragma: no cover
    """Build MatMulFpQ4 node.

    Args:
        node: original matmul node
        weight_shape: original weight shape
        num_bits (int): num_bits
        group_size (int): how many elements share one scale/zp
        k_blocks (int): block number
        q_weight (array): quantized weight
        scale (array): scale
        zero_point (array): zero point
        accuracy_level (int): accuracy level. Support 0 (unset), 1(fp32 compute type of jblas kernel),
                              2 (fp16 compute type of jblas kernel), 3 (bf16 compute type of jblas kernel),
                              4 (int8 compute type of jblas kernel)

    Returns:
        matmul_weight_only_node: MatMulFpQ4 or MatMulNBits node
        new_inits: initializers of the new node
    """
    blob_size = get_blob_size(group_size, zero_point is not None)
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
