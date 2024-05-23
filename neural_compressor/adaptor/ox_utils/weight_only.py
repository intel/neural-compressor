#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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
"""WeightOnly for onnxrt adaptor."""

import copy
import logging
import math
import os
import struct
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnx import onnx_pb as onnx_proto
from packaging.version import Version

from neural_compressor.adaptor.ox_utils.util import dtype_mapping, simple_progress_bar
from neural_compressor.model.model import BaseModel
from neural_compressor.model.onnx_model import ONNXModel
from neural_compressor.utils.utility import LazyImport

ort = LazyImport("onnxruntime")
logger = logging.getLogger("neural_compressor")
ONNXRT116_VERSION = Version("1.16.0")
ONNXRT1161_VERSION = Version("1.16.1")


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
        q_weight_pairs = q_weight[:, ::2] | q_weight[:, 1::2] << 4
        packed[:, :] = q_weight_pairs[:, :blob_size]
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
                # create an index array
                idx = np.arange(zero_point.shape[0] // k_blocks * k_blocks).reshape(-1)
                # separate odd and even indices
                even_idx = idx[::2]
                odd_idx = idx[1::2]
                # vectorized operation for even and odd indices
                packed_zp[even_idx // 2] = (packed_zp[even_idx // 2] & 0xF0) | zero_point[even_idx].ravel()
                packed_zp[odd_idx // 2] = (packed_zp[odd_idx // 2] & 0x0F) | (zero_point[odd_idx].ravel() << 4)

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
        mask = max_range > 0
        scale[mask] = (max_range[mask] * 2.0).astype(np.float64) / (maxq - minq)
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

    q_weight = np.empty_like(data, dtype=scale.dtype)
    np.divide(data, scale, out=q_weight)
    np.add(q_weight, zero_point, out=q_weight)
    np.round(q_weight, out=q_weight)
    np.clip(q_weight, minq, maxq, out=q_weight)

    return q_weight, scale, zero_point


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


def rtn_quantize(
    model,
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
        model (ModelProto or ONNXModel): onnx model
        weight_config (dict): quantization config
                For example,
                weight_config = {
                    'fc2':
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
    model = model if isinstance(model, BaseModel) else ONNXModel(model)
    base_dir = os.path.dirname(model.model_path) if model.model_path is not None else ""
    new_nodes = []
    remove_nodes = []
    total_num = len([i for i in model.nodes() if i.op_type in ["MatMul"]])
    curr_id = 0
    for node in model.nodes():
        if node.op_type in ["MatMul"]:
            curr_id += 1
            simple_progress_bar(total_num, curr_id)
        if (
            node.op_type in ["MatMul"]
            and model.get_initializer(node.input[1]) is not None
            and weight_config.get(node.name, {}) != "fp32"
        ):
            weight_tensor = model.get_initializer(node.input[1])
            weight = numpy_helper.to_array(weight_tensor, base_dir=base_dir).copy()
            if len(weight.shape) != 2:
                continue

            dtype = weight.dtype

            if node.name in weight_config:
                num_bits = weight_config[node.name]["bits"]
                group_size = weight_config[node.name]["group_size"]
                scheme = weight_config[node.name]["scheme"]

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
    return model


def get_weight_scale(weight, group_size):
    """Get the scale of weight."""
    org_shape = weight.shape
    weight = np.reshape(weight, (-1, group_size)) if group_size != -1 else weight
    scale = np.mean(np.reshape(np.abs(weight) / np.max(np.abs(weight), axis=1, keepdims=True), org_shape), axis=0)
    return scale


def apply_awq_scale(model, weight_config, absorb_pairs, output_dicts, num_bits, group_size, scheme):
    """Apply scale for salient weight."""
    best_scales = {}
    new_init_tensors = []
    new_added_mul_nodes = []
    replace_input = []
    updated_nodes = []
    base_dir = os.path.dirname(model.model_path) if model.model_path is not None else ""

    for parent, nodes in absorb_pairs.items():
        if any([node.input[0] not in output_dicts for node in nodes]):
            logger.warning(
                "Miss input tensors of nodes {} during AWQ, skip it!".format(
                    ", ".join([node.name for node in nodes if node.input[0] not in output_dicts])
                )
            )
            continue
        inp = np.concatenate(output_dicts[nodes[0].input[0]], axis=0)
        inp_scale = np.mean(np.reshape(np.abs(inp), (-1, inp[0].shape[-1])), axis=0)
        dtype = None
        weight = []
        org_out = []
        for node in nodes:
            if node.name in weight_config and weight_config.get(node.name, "fp32") != "fp32":
                num_bits = weight_config[node.name]["bits"]
                group_size = weight_config[node.name]["group_size"]
                scheme = weight_config[node.name]["scheme"]
                break

        # search scale
        best_error = float("inf")
        best_ratio = -1
        best_scale = None
        n_grid = 20

        for ratio in range(n_grid):
            ratio = ratio * 1 / n_grid
            loss = 0
            for node in nodes:
                if weight_config.get(node.name, {}) == "fp32":
                    continue

                weight = numpy_helper.to_array(model.get_initializer(node.input[1]), base_dir)
                if len(weight.shape) != 2:
                    continue

                org_out = np.matmul(inp, weight)
                org_w_shape = weight.shape
                group_size = group_size if group_size != -1 else org_w_shape[0]

                w_scale = get_weight_scale(weight.T, weight.shape[0])
                scales = np.clip(np.power(inp_scale, ratio) / np.power(w_scale, (1 - ratio)), 1e-4, None)
                scales = scales / np.sqrt(np.max(scales) * np.min(scales))
                weight = weight.T * scales
                weight = pad_tensor(weight.T, group_size, (org_w_shape[0] + group_size - 1) // group_size)

                if (Version(ort.__version__) > ONNXRT1161_VERSION and num_bits == 4) or (
                    Version(ort.__version__) >= ONNXRT116_VERSION and num_bits == 4 and group_size == 32
                ):  # pragma: no cover
                    # MatMulFpQ4 support 4 bits and 32 group_size with ort 1.16.0 and 1.16.1 versions
                    # MatMulNBits supports 4 bits and 2^n group_size with ort > 1.16.1
                    q_weight = qdq_tensor(weight, num_bits, group_size, scheme, "uint") / np.expand_dims(
                        scales, axis=-1
                    )
                else:
                    q_weight = qdq_tensor(weight, num_bits, group_size, scheme, "int") / np.expand_dims(scales, axis=-1)

                q_weight = np.reshape(q_weight, (org_w_shape[1], -1))[:, : org_w_shape[0]]

                out = np.matmul(inp, q_weight.T)
                loss += np.mean(np.power((org_out - out), 2))

            is_best = loss < best_error
            if is_best:
                best_error = loss
                best_ratio = ratio
                best_scale = scales

        for node in nodes:
            weight_config.setdefault(node.name, {}).update({"bits": num_bits})
            weight_config.setdefault(node.name, {}).update({"group_size": group_size})
            weight_config.setdefault(node.name, {}).update({"scheme": scheme})

            init_share_num = model.get_initializer_share_num(node.input[1])
            weight_tensor = model.get_initializer(node.input[1])
            tensor = numpy_helper.to_array(weight_tensor, base_dir)
            dtype = tensor.dtype
            tensor = tensor.T * best_scale
            tensor = (tensor.T).astype(dtype)

            new_tensor = onnx.helper.make_tensor(
                name=node.input[1] + "_scaled",
                data_type=dtype_mapping[str(dtype)],
                dims=tensor.shape,
                vals=tensor.tobytes(),
                raw=True,
            )
            model.add_initializer(new_tensor)
            node.input[1] = new_tensor.name

            if init_share_num == 1:
                model.remove_initializer(weight_tensor)

        parent = model.get_node(parent)
        if parent.name in updated_nodes:
            continue

        if parent.op_type in ["LayerNormalization", "BatchNormalization", "InstanceNormalization"] and len(
            model.input_name_to_nodes[nodes[0].input[0]]
        ) == len(nodes):
            for idx in [1, 2]:
                tensor = numpy_helper.to_array(model.get_initializer(parent.input[idx]), base_dir)
                dtype = tensor.dtype
                new_tensor = tensor / np.reshape(best_scale, (1, -1))
                model.set_initializer(parent.input[idx], new_tensor.astype(dtype), raw=True)
                updated_nodes.append(parent.name)
            output_dicts[parent.output[0]] = output_dicts[parent.output[0]] / np.reshape(best_scale, (1, -1))

        elif (
            parent.op_type in ["SimplifiedLayerNormalization", "MatMul", "Gemm", "Mul"]
            and not all([model.get_initializer(inp) is None for inp in parent.input])
            and len(model.input_name_to_nodes[nodes[0].input[0]]) == len(nodes)
        ):  # pragma: no cover
            for inp in parent.input:
                if model.get_initializer(inp) is not None:
                    tensor = numpy_helper.to_array(model.get_initializer(inp), base_dir)
                    dtype = tensor.dtype
                    new_tensor = tensor / np.reshape(best_scale, (1, -1))
                    model.set_initializer(inp, new_tensor.astype(dtype), raw=True)
            updated_nodes.append(parent.name)
            output_dicts[parent.output[0]] = output_dicts[parent.output[0]] / np.reshape(best_scale, (1, -1))

        elif parent.op_type in ["Conv", "FusedConv"] and len(model.input_name_to_nodes[nodes[0].input[0]]) == len(
            nodes
        ):  # pragma: no cover
            tensor = numpy_helper.to_array(model.get_initializer(parent.input[2]), base_dir)
            dtype = tensor.dtype
            new_tensor = tensor / np.reshape(best_scale, (1, -1))
            model.set_initializer(parent.input[2], new_tensor.astype(dtype), raw=True)
            updated_nodes.append(parent.name)
            output_dicts[parent.output[0]] = output_dicts[parent.output[0]] / np.reshape(best_scale, (1, -1))

        else:  # pragma: no cover
            # insert mul
            scale_tensor = helper.make_tensor(
                name=parent.output[0] + "_weight_only_scale",
                data_type=dtype_mapping[str(dtype)],
                dims=best_scale.shape,
                vals=(1.0 / best_scale).flatten().tolist(),
            )
            new_init_tensors.append(scale_tensor)
            mul_output_name = parent.output[0] + "_weight_only_out"
            mul_node = helper.make_node(
                "Mul",
                inputs=[nodes[0].input[0], scale_tensor.name],
                outputs=[mul_output_name],
                name=nodes[0].input[0] + "_weight_only_mul",
            )
            new_added_mul_nodes.append(mul_node)
            for node in nodes:
                replace_input.append([node, node.input[0], mul_node.output[0]])
            updated_nodes.append(parent.name)
            output_dicts[mul_node.output[0]] = output_dicts[mul_node.input[0]] / np.reshape(best_scale, (1, -1))

    model.add_nodes(new_added_mul_nodes)
    model.add_initializers(new_init_tensors)
    for node, old_input_name, new_input_name in replace_input:
        model.replace_node_input(node, old_input_name, new_input_name)

    return model, output_dicts


def apply_awq_clip(model, weight_config, absorb_pairs, output_dicts, num_bits, group_size, scheme):
    """Apply clip for weight by checking mse."""
    base_dir = os.path.dirname(model.model_path) if model.model_path is not None else ""
    ratios = {}
    for parent, nodes in absorb_pairs.items():
        if any([node.input[0] not in output_dicts for node in nodes]):
            logger.warning(
                "Miss input tensors of nodes {} during AWQ, skip it!".format(
                    ", ".join([node.name for node in nodes if node.input[0] not in output_dicts])
                )
            )
            continue

        inp = np.concatenate(output_dicts[nodes[0].input[0]], axis=0)

        for node in nodes:
            if node.name in weight_config:
                num_bits = weight_config[node.name]["bits"]
                group_size = weight_config[node.name]["group_size"]
                scheme = weight_config[node.name]["scheme"]

            org_weight = numpy_helper.to_array(model.get_initializer(node.input[1]), base_dir=base_dir)
            org_w_shape = org_weight.shape  # ic, oc
            group_size = group_size if group_size != -1 else org_w_shape[0]
            org_out = np.matmul(inp, org_weight)  # n_token, oc

            k_blocks = (org_w_shape[0] - 1) // group_size + 1
            org_weight = pad_tensor(org_weight, group_size, k_blocks)

            org_weight = np.transpose(org_weight)

            best_error = float("inf")
            best_ratio = 1
            for i_s in range(10):
                ratio = 1 - i_s / 100
                weight = copy.deepcopy(org_weight)
                if (Version(ort.__version__) > ONNXRT1161_VERSION and num_bits == 4) or (
                    Version(ort.__version__) >= ONNXRT116_VERSION and num_bits == 4 and group_size == 32
                ):  # pragma: no cover
                    # MatMulFpQ4 support 4 bits and 32 group_size with ort 1.16.0 and 1.16.1 versions
                    # MatMulNBits supports 4 bits and 2^n group_size with ort > 1.16.1
                    weight = qdq_tensor(weight, num_bits, group_size, scheme, "uint", ratios.get(node.input[1], 1))
                else:
                    weight = qdq_tensor(weight, num_bits, group_size, scheme, "int", ratios.get(node.input[1], 1))
                weight = np.reshape(weight, (org_w_shape[1], -1))[:, : org_w_shape[0]]
                cur_out = np.matmul(inp, weight.T)
                loss = np.mean(np.power((org_out - cur_out), 2))
                is_best = loss < best_error
                if is_best:
                    best_error = loss
                    best_ratio = ratio
            ratios[node.input[1]] = best_ratio
    return ratios


def prepare_inputs(model, n_samples, dataloader, providers):
    """Prepare inputs for weight only quantization.

    Args:
        model (ModelProto or ONNXModel): onnx model
        n_samples (int, optional): calibration sample number. -1 means all samples.
        dataloader (object): dataloader for calibration.
        providers (list): providers to use

    Returns:
        inputs: prepared inputs.
        so: session options
    """
    from importlib.util import find_spec

    from neural_compressor.adaptor.ox_utils.util import to_numpy

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

    session = (
        ort.InferenceSession(model.model.SerializeToString(), so, providers=providers)
        if not model.is_large_model
        else ort.InferenceSession(model.model_path + "_augment.onnx", so, providers=providers)
    )
    inputs_names = [i.name for i in session.get_inputs()]
    del session

    inputs = []
    for i, data in enumerate(dataloader):
        if n_samples != -1 and ((i + 1) * dataloader.batch_size) > n_samples:
            break
        if len(inputs_names) != 1 or isinstance(data[0], dict):
            assert len(data[0]) == len(inputs_names), "Input number mismatch, " "require {} but get {}".format(
                len(inputs_names), len(data[0])
            )

        if isinstance(data[0], dict):
            inputs.append(dict([(name, to_numpy(inp_data)) for name, inp_data in data[0].items()]))
        elif isinstance(data[0], np.ndarray):  # pragma: no cover
            inputs.append(dict([(name, inp) for name, inp in zip(inputs_names, [data[0]])]))
        else:  # pragma: no cover
            inputs.append(dict([(name, to_numpy(inp)) for name, inp in zip(inputs_names, data[0])]))
    return inputs, so


def awq_quantize(
    model,
    dataloader,
    weight_config={},
    num_bits=4,
    group_size=32,
    scheme="asym",
    n_samples=128,
    enable_auto_scale=True,
    enable_mse_search=True,
    accuracy_level=0,
    providers=["CPUExecutionProvider"],
):
    """Quant the model with Activation-aware Weight quantization(AWQ) method.

    Args:
        model (ModelProto or ONNXModel): onnx model
        dataloader (object): dataloader for calibration.
        weight_config (dict): quantization config
                For example,
                weight_config = {
                    'fc2':
                        {
                            'bits': 4,
                            'group_size': 32,
                            'scheme': 'sym',
                            'algorithm': 'AWQ'
                        }
                }
        num_bits (int, optional): num_bits. Default is 4.
        group_size (int, optional): how many elements share one scale/zp. Default is 32.
        scheme (str, optional): sym or asym. Defaults to "asym".
        n_samples (int, optional): calibration sample number.
        enable_auto_scale (bool, optional): whether enable scale for salient weight. Defaults to True.
        enable_mse_search (bool, optional):  whether enable clip for weight by checking mse. Defaults to True.
        accuracy_level (int): accuracy level. Support 0 (unset), 1(fp32 compute type of jblas kernel),
                              2 (fp16 compute type of jblas kernel), 3 (bf16 compute type of jblas kernel),
                              4 (int8 compute type of jblas kernel)
        providers (list): providers to use

    Returns:
        model: fake quantized ONNXModel
    """
    model = model if isinstance(model, BaseModel) else ONNXModel(model)
    output_dicts = {}
    full_ratio = {}

    if enable_mse_search:
        inputs, so = prepare_inputs(model, n_samples, dataloader, providers)
        del dataloader

        org_output = copy.deepcopy(model.model.graph.output)
        model.remove_tensors_from_outputs([i.name for i in org_output])

        output_names = []

        for node in model.nodes():
            if (
                node.op_type in ["MatMul"]
                and weight_config.get(node.name, {}) != "fp32"
                and weight_config.get(node.name, {}).get("algorithm", "AWQ") == "AWQ"
            ):
                output_names.append(node.input[0])
        output_names = list(set(output_names))
        model.add_tensors_to_outputs(output_names)
        if model.is_large_model:
            onnx.save_model(
                model.model,
                model.model_path + "_augment.onnx",
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                convert_attribute=False,
            )

        session = (
            ort.InferenceSession(model.model.SerializeToString(), so, providers=providers)
            if not model.is_large_model
            else ort.InferenceSession(model.model_path + "_augment.onnx", so, providers=providers)
        )

        for input_name in output_names:
            parent = model.output_name_to_node[input_name]
            dump_pairs = {parent.name: []}

            for node in model.input_name_to_nodes[input_name]:
                if (
                    node.op_type in ["MatMul"]
                    and weight_config.get(node.name, {}) != "fp32"
                    and weight_config.get(node.name, {}).get("algorithm", "AWQ") == "AWQ"
                    and model.get_initializer(node.input[1]) is not None
                ):
                    dump_pairs[parent.name].append(model.get_node(node.name))

            if len(dump_pairs[parent.name]) == 0:
                continue

            output_dicts = {}
            for inp in inputs:
                output = session.run([input_name], inp)
                output_dicts.setdefault(input_name, []).append(output)

            if enable_auto_scale:
                model, output_dicts = apply_awq_scale(
                    model,
                    weight_config,
                    dump_pairs,
                    output_dicts,
                    num_bits,
                    group_size,
                    scheme,
                )
            if enable_mse_search:
                ratios = apply_awq_clip(
                    model,
                    weight_config,
                    dump_pairs,
                    output_dicts,
                    num_bits,
                    group_size,
                    scheme,
                )
            del output_dicts
            del dump_pairs
            full_ratio.update(ratios)

        model.remove_tensors_from_outputs(output_names)
        model.model.graph.output.MergeFrom(org_output)
    model = rtn_quantize(model, weight_config, num_bits, group_size, scheme, full_ratio, accuracy_level, providers)
    return model


def gptq(
    W,
    H,
    num_bits=4,
    group_size=32,
    scheme="asym",
    blocksize=128,
    percdamp=0.01,
    actorder=False,
    mse=False,
    perchannel=True,
):
    """Quant the weight with GPTQ method.

    Args:
        W (array): weight.
        H (array): Hessian matrix.
        num_bits (int, optional): num_bits. Default is 4.
        group_size (int, optional): how many elements share one scale/zp. Default is 32.
        scheme (str, optional): sym or asym. Defaults to "asym".
        blocksize (int, optional): blocksize to quantize weight.
        percdamp (float, optional): percent of the average Hessian diagonal to use for dampening.
        actorder (bool, optional): whether rearrange Hessian matrix considering the diag's value.
        mse (bool, optional): whether get scale and zero point with mse error.
        perchannel (bool, optional): whether quantize weight per-channel.

    Returns:
        Q: fake quantized weight
    """
    Qs = []
    maxq = 2**num_bits - 1
    grid = 100
    maxshrink = 0.8
    norm = 2.4

    def find_params(weight):
        org_shape = weight.shape
        # find zp, scale
        if not perchannel:
            weight = np.expand_dims(weight.flatten(), axis=1)
        tmp = np.zeros(weight.shape[1])
        xmin = np.minimum(np.min(weight, axis=0), tmp)
        xmax = np.maximum(np.max(weight, axis=0), tmp)
        if scheme == "sym":
            xmax = np.maximum(np.abs(xmin), xmax)
            tmp = xmin < 0
            if np.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        scale = (xmax - xmin) / maxq
        if scheme == "sym":
            zero = np.ones(scale.shape) * (maxq + 1) / 2
        else:
            zero = np.round(-xmin / scale)
        if mse:
            best = np.ones([weight.shape[1]]) * float("inf")
            for i in range(int(maxshrink * grid)):
                p = 1 - i / grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / maxq
                zero1 = np.round(-xmin1 / scale1) if scheme != "sym" else zero
                q = np.clip(np.round(weight / scale1) + zero1, 0, maxq)
                q -= weight
                q = np.power(np.abs(q), norm)
                err = np.sum(q, 0)
                tmp = err < best
                if np.any(tmp):
                    best[tmp] = err[tmp]
                    scale[tmp] = scale1[tmp]
                    zero[tmp] = zero1[tmp]
        if not perchannel:
            tmp = org_shape[1]
            scale = np.repeat(scale, tmp)
            zero = np.repeat(zero, tmp)
        shape = [-1] + [1] * (len(org_shape) - 1)
        scale = np.reshape(scale, shape)
        zero = np.reshape(zero, shape)
        return scale, zero

    scales = []
    zps = []
    shape = W.shape
    scale, zp = find_params(W)
    dead = np.diag(H) == 0
    H[dead, dead] = 1
    W[dead, :] = 0  # such channel makes no contribution to quantization computation

    # rearrange considering the diag's value
    if actorder:
        perm = np.argsort(np.diag(H))[::-1]
        W = W[perm, :]
        H = H[perm, :][:, perm]
    Losses = np.zeros_like(W)
    Q = np.zeros_like(W)
    damp = percdamp * np.mean(np.diag(H))
    diag = np.arange(shape[0])
    H[diag, diag] += damp  # add a average value of
    H = np.linalg.cholesky(np.linalg.inv(H)).T
    Hinv = H
    for i1 in range(0, shape[0], blocksize):
        i2 = min(i1 + blocksize, shape[0])
        count = i2 - i1

        W1 = copy.deepcopy(W[i1:i2, :])
        Q1 = np.zeros_like(W1)
        Err1 = np.zeros_like(W1)
        Losses1 = np.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        for i in range(count):  # within a block, channel wise
            w = W1[i, :]
            d = Hinv1[i, i]

            if group_size != -1:
                if (i1 + i) % group_size == 0:
                    scale, zp = find_params(W[(i1 + i) : (i1 + i + group_size), :])

            q = (scale * (np.clip(np.round(w[:, np.newaxis] / scale) + zp, 0, maxq) - zp)).flatten()
            Q1[i, :] = q
            Losses1[i, :] = (w - q) ** 2 / d**2

            err1 = (w - q) / d
            W1[i:, :] -= np.matmul(np.expand_dims(Hinv1[i:, i], axis=1), np.expand_dims(err1, axis=0))
            Err1[i, :] = err1

        Q[i1:i2, :] = Q1
        Losses[i1:i2, :] = Losses1 / 2

        W[i2:, :] -= np.matmul(Hinv[i2:, i1:i2], Err1)

    if actorder:
        invperm = np.argsort(perm)
        Q = Q[invperm, :]

    Q = np.reshape(Q, W.shape)
    del W
    return Q


def gptq_quantize(
    model,
    dataloader,
    weight_config={},
    num_bits=4,
    group_size=32,
    scheme="asym",
    n_samples=128,
    percdamp=0.01,
    blocksize=128,
    actorder=False,
    mse=False,
    perchannel=True,
    accuracy_level=0,
    providers=["CPUExecutionProvider"],
):
    """Quant the model with GPTQ method.

    Args:
        model (ModelProto or ONNXModel): onnx model
        dataloader (object): dataloader for calibration.
        weight_config (dict): quantization config
                For example,
                weight_config = {
                    'fc2':
                        {
                            'bits': 4,
                            'group_size': 32,
                            'scheme': 'sym',
                            'algorithm': 'GPTQ'
                        }
                }
        num_bits (int, optional): num_bits. Default is 4.
        group_size (int, optional): how many elements share one scale/zp. Default is 32.
        scheme (str, optional): sym or asym. Defaults to "asym".
        n_samples (int, optional): calibration sample number.
        percdamp (float, optional): percent of the average Hessian diagonal to use for dampening.
        blocksize (int, optional): blocksize to quantize weight.
        actorder (bool, optional): whether rearrange Hessian matrix considering the diag's value.
        mse (bool, optional): whether get scale and zero point with mse error.
        perchannel (bool, optional): whether quantize weight per-channel.
        accuracy_level (int): accuracy level. Support 0 (unset), 1(fp32 compute type of jblas kernel),
                              2 (fp16 compute type of jblas kernel), 3 (bf16 compute type of jblas kernel),
                              4 (int8 compute type of jblas kernel)
        providers (list): providers to use

    Returns:
        model: fake quantized ONNXModel
    """
    model = model if isinstance(model, BaseModel) else ONNXModel(model)
    base_dir = os.path.dirname(model.model_path) if model.model_path is not None else ""
    output_dicts = {}

    inputs, so = prepare_inputs(model, n_samples, dataloader, providers)
    del dataloader
    org_output = copy.deepcopy(model.model.graph.output)
    model.remove_tensors_from_outputs([i.name for i in org_output])
    output_names = []
    for node in model.nodes():
        if (
            node.op_type in ["MatMul"]
            and weight_config.get(node.name, {}) != "fp32"
            and weight_config.get(node.name, {}).get("algorithm", "GPTQ") == "GPTQ"
        ):
            output_names.append(node.input[0])
    output_names = list(set(output_names))
    model.add_tensors_to_outputs(output_names)
    if model.is_large_model:
        onnx.save_model(
            model.model,
            model.model_path + "_augment.onnx",
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            convert_attribute=False,
        )

    session = (
        ort.InferenceSession(model.model.SerializeToString(), so, providers=providers)
        if not model.is_large_model
        else ort.InferenceSession(model.model_path + "_augment.onnx", so, providers=providers)
    )

    new_nodes = []
    remove_nodes = []
    for idx, input_name in enumerate(output_names):
        simple_progress_bar(len(output_names), idx + 1)
        node_list = []
        weights = []

        for node in model.input_name_to_nodes[input_name]:
            if (
                node.op_type in ["MatMul"]
                and weight_config.get(node.name, {}) != "fp32"
                and weight_config.get(node.name, {}).get("algorithm", "GPTQ") == "GPTQ"
                and model.get_initializer(node.input[1]) is not None
            ):
                weight = numpy_helper.to_array(
                    model.get_initializer(model.get_node(node.name).input[1]), base_dir
                ).copy()
                if len(weight.shape) != 2:
                    continue

                weights.append(weight)
                node_list.append(model.get_node(node.name))

        if len(weights) == 0:
            continue

        Hs = [np.zeros((i.shape[0], i.shape[0])) for i in weights]
        nsamples = 0
        for data in inputs:
            inp = session.run([input_name], data)[0]
            tmp = inp.shape[0]
            inp = np.reshape(inp, (-1, inp.shape[-1]))
            Hs = [i * (nsamples / (nsamples + tmp)) for i in Hs]
            nsamples += tmp
            inp = np.sqrt(2 / nsamples) * inp
            Hs = [i + np.matmul(inp.T, inp) for i in Hs]

        for (
            node,
            weight,
            H,
        ) in zip(node_list, weights, Hs):
            if node.name in weight_config:
                num_bits = weight_config[node.name]["bits"]
                group_size = weight_config[node.name]["group_size"]
                scheme = weight_config[node.name]["scheme"]
            group_size = group_size if group_size != -1 else weight.shape[0]
            dtype = weight.dtype

            q_weight = gptq(
                weight,
                H,
                num_bits=num_bits,
                group_size=group_size,
                scheme=scheme,
                blocksize=blocksize,
                percdamp=percdamp,
                actorder=actorder,
                mse=mse,
                perchannel=perchannel,
            )

            weight_tensor = model.get_initializer(node.input[1])
            init_share_num = model.get_initializer_share_num(node.input[1])

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
                org_shape = weight.shape
                k_blocks = (org_shape[0] + group_size - 1) // group_size
                q_weight = pad_tensor(q_weight, group_size, k_blocks)
                q_weight, scale, zp = quant_tensor(q_weight.T, num_bits, group_size, scheme, "uint")
                q_matmul_node, new_inits = make_matmul_weight_only_node(
                    node=node,
                    weight_shape=org_shape,
                    num_bits=num_bits,
                    group_size=group_size,
                    k_blocks=k_blocks,
                    q_weight=q_weight.astype("uint8"),
                    scale=scale.astype(dtype),
                    zero_point=zp if scheme == "asym" else None,
                    accuracy_level=accuracy_level,
                )

                model.add_initializers(new_inits)
                model.remove_node(node)
                model.add_node(q_matmul_node)
            else:
                q_weight_tensor = onnx.helper.make_tensor(
                    name=node.input[1] + "_Q{}G{}".format(str(num_bits), str(group_size)),
                    data_type=dtype_mapping[str(dtype)],
                    dims=q_weight.shape,
                    vals=q_weight.astype(dtype).tobytes(),
                    raw=True,
                )
                model.add_initializer(q_weight_tensor)
                node.input[1] = q_weight_tensor.name
            if init_share_num == 1:
                model.remove_initializer(weight_tensor)

    model.remove_tensors_from_outputs(output_names)
    model.model.graph.output.MergeFrom(org_output)

    model.topological_sort()

    # reload external data to prevent external data file path errors
    if model.is_large_model:
        from onnx.external_data_helper import load_external_data_for_model

        load_external_data_for_model(model.model, os.path.split(model.model_path)[0])

    return model
