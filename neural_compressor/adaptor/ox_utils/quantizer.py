#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
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
#
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Quantizer for onnx models."""

import copy
import logging
import os

import numpy as np
import onnx
from onnx import TensorProto
from onnx import onnx_pb as onnx_proto
from onnx import shape_inference
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions

from neural_compressor.adaptor.ox_utils.operators import OPERATORS
from neural_compressor.adaptor.ox_utils.util import (
    QuantizedInitializer,
    QuantizedValue,
    QuantizedValueType,
    ValueInfo,
    __producer__,
    __version__,
    _get_qrange_for_qType,
    cast_tensor,
    dtype_mapping,
    dtype_to_name,
    find_by_name,
    get_node_original_name,
    make_dquant_node,
    make_quant_node,
    quantize_data,
    quantize_data_per_channel,
    support_pair,
)
from neural_compressor.model.onnx_model import ONNXModel

logger = logging.getLogger("neural_compressor")


class Quantizer:
    """Quantizer class."""

    def __init__(
        self,
        model,
        q_config,
        mode,
        static,
        quantization_params,
        op_types_to_quantize,
        fallback_list=["fp32"],
        reduce_range=None,
        add_qdq_pair_to_weight=False,
        optypes_to_exclude_output_quant=[],
        dedicated_qdq_pair=False,
        backend="CPUExecutionProvider",
    ):
        """Initialization.

        Args:
            model (ModelProto or ONNXModel): onnx model or onnx model wrapper by neural compressor
            q_config (dict): op-wise quantization config.
            mode (QuantizationMode): quantizaion mode
            static (bool): static or not
            quantization_params (dict): scale and zero point of tensors
            op_types_to_quantize (list): optypes to quantize
            fallback_list (list, optional): fallback data type. Defaults to ['fp32'].
            reduce_range (bool, optional): use 7 bit or not. Defaults to None.
            add_qdq_pair_to_weight (bool, optional): add QDQ pair to weight or not. Defaults to False.
            optypes_to_exclude_output_quant (list, optional): optypes to exclude output quantization. Defaults to [].
            dedicated_qdq_pair (bool, optional): dedicate QDQ pair or not. Defaults to False.
            backend (str, optional): backend of onnxrt adaptor. Defaults to CPUExecutionProvider
        """
        self.model = ONNXModel(model) if not isinstance(model, ONNXModel) else model
        model = (
            onnx.shape_inference.infer_shapes(self.model.model) if not self.model.is_large_model else self.model.model
        )
        self.config = q_config
        self.backend = backend
        self.reduce_range = reduce_range
        self.mode = mode  # QuantizationMode.Value
        self.static = static  # use static quantization for inputs.
        self.fuse_dynamic_quant = False
        self.quantization_params = quantization_params
        self.op_types_to_quantize = op_types_to_quantize
        self.fallback_list = fallback_list
        self.new_nodes = []

        self.opset_version = self.check_opset_version()
        self.value_infos = {vi.name: vi for vi in model.graph.value_info}
        self.value_infos.update({ot.name: ot for ot in model.graph.output})
        self.value_infos.update({it.name: it for it in model.graph.input})
        self.replace_input = []
        self.remove_nodes = []
        # List of quantized weights
        self.quantized_value_map = {}
        self.new_value_info = {}

        # List of recalculated quantize weight for Gather op.
        self.recalculate_quantized_value = []

        # QuantizeRange tensor name and zero tensor name for scale and zero point calculation.
        # Used when static is False
        self.fixed_qrange_uint8_name = "fixed_quantization_range_uint8"
        self.fixed_qrange_int8_name = "fixed_quantization_range_int8"
        # For uint8 data-type, to compute zero point, we subtract rmin from 0 (represented by fixed_zero_name tensor)
        self.fixed_zero_name = "fixed_zero"
        # For int8 data-type, zero point is always zero (represented by fixed_zero_point_name tensor)
        self.fixed_zero_zp_name = "fixed_zero_zp"

        if not self.static:
            self.op_types_to_exclude_output_quantization = op_types_to_quantize
        else:
            self.op_types_to_exclude_output_quantization = optypes_to_exclude_output_quant

        self.add_qdq_pair_to_weight = add_qdq_pair_to_weight
        self.dedicated_qdq_pair = dedicated_qdq_pair

    def check_opset_version(self):
        """Check opset version."""
        ai_onnx_domain = [
            opset for opset in self.model.model.opset_import if not opset.domain or opset.domain == "ai.onnx"
        ]
        if 1 != len(ai_onnx_domain):
            raise ValueError("Failed to find proper ai.onnx domain")
        opset_version = ai_onnx_domain[0].version

        if opset_version > 10:
            self.fuse_dynamic_quant = True
        elif opset_version < 10:
            logger.warning(
                f"Warning: The original model opset version is {opset_version}, which does not support node "
                + "fusions. Please update the model to opset >= 11 for better performance."
            )
            self.model.model.opset_import.remove(ai_onnx_domain[0])
            self.model.model.opset_import.extend([onnx.helper.make_opsetid("", 11)])
            opset_version = 11

        if opset_version < 13 and self.mode == "qdq":
            logger.warning(
                "Per-channel support with QDQ format requires opset version >= 13,"
                " use per-tensor granularity instead"
            )
        return opset_version

    def should_quantize(self, node):
        """Check if node should be quantized."""
        if node.name in self.config and self.config[node.name] not in self.fallback_list:
            return True
        elif (
            get_node_original_name(node) in self.config
            and self.config[get_node_original_name(node)] not in self.fallback_list
        ):
            return True
        else:
            return False

    def quantize_model(self):
        """Quantize onnx model."""
        # step 1: insert q-dq, cast-cast pairs
        self.insert_qdq()

        # step 2: remove redundant pairs -> qdq model
        self.remove_redundant_pairs()

        # step 3: convert q-node-dq to qlinear op if needed
        self.convert_qdq_to_operator_oriented()

        self.merge_dedicated_qdq_pair()

        self.model.remove_unused_nodes()

        self.model.model.producer_name = __producer__
        self.model.model.producer_version = __version__

        return self.model.model

    def merge_dedicated_qdq_pair(self):
        """Merge dedicated Q/DQ pairs."""
        self.remove_nodes = []
        self.replace_input = []
        self.new_nodes = []
        if self.mode == "qdq" and self.dedicated_qdq_pair:
            for node in self.model.nodes():
                if node.op_type in ["QuantizeLinear"]:
                    children = self.model.get_children(node)
                    if len([i for i in children if i.op_type in ["DequantizeLinear"]]) < 2:
                        continue
                    for idx, child in enumerate(children):
                        if child.op_type not in ["DequantizeLinear"]:
                            continue
                        if self.should_quantize(self.model.get_children(child)[0]):
                            inputs = [self.model.get_parents(node)[0].output[0], node.input[1], node.input[2]]
                            self.new_nodes.append(
                                onnx.helper.make_node(
                                    "QuantizeLinear",
                                    inputs,
                                    [node.output[0] + "_" + str(idx)],
                                    node.name + "_" + str(idx),
                                )
                            )
                            self.replace_input.append([child, node.output[0], node.output[0] + "_" + str(idx)])
                        else:
                            self.remove_nodes.append(child)
                            self.replace_input.append(
                                [self.model.get_children(child)[0], child.output[0], node.input[0]]
                            )
                    self.remove_nodes.append(node)
            self.model.remove_nodes(self.remove_nodes)
            self.model.graph().node.extend(self.new_nodes)
            for node, old_input_name, new_input_name in self.replace_input:
                self.model.replace_node_input(node, old_input_name, new_input_name)
            self.model.update()
        elif self.mode != "qdq" or not self.dedicated_qdq_pair:
            target_type = ["QuantizeLinear", "DequantizeLinear"]
            for op_type in target_type:
                for node in self.model.nodes():
                    children = self.model.get_children(node)
                    dq_nodes = [i for i in children if i.op_type == op_type]
                    if len(dq_nodes) < 2 or node.op_type in ["Split"]:
                        continue
                    datas = []
                    for n in dq_nodes:
                        datas.append(
                            [
                                onnx.numpy_helper.to_array(find_by_name(n.input[1], self.model.initializer())),
                                onnx.numpy_helper.to_array(find_by_name(n.input[2], self.model.initializer())),
                            ]
                        )
                    for idx, data in enumerate(datas):
                        repeaded_id = [i for i, item in enumerate(datas[idx:]) if item == data]
                        for i in repeaded_id[1:]:
                            self.remove_nodes.append(dq_nodes[i])
                            self.replace_input.append(
                                [
                                    self.model.get_children(dq_nodes[i])[0],
                                    dq_nodes[i].output[0],
                                    dq_nodes[idx].output[0],
                                ]
                            )
                self.model.remove_nodes(self.remove_nodes)
                self.model.graph().node.extend(self.new_nodes)
                for node, old_input_name, new_input_name in self.replace_input:
                    self.model.replace_node_input(node, old_input_name, new_input_name)
                self.model.update()

        if self.mode == "qdq":
            for node in self.model.nodes():
                if node.op_type in ["QuantizeLinear"] and len(self.model.get_parents(node)) > 0:
                    if "QuantizeLinear" in [sibling.op_type for sibling in self.model.get_siblings(node)]:
                        continue
                    for sibling in self.model.get_siblings(node):
                        if not self.should_quantize(sibling) and sibling.op_type in OPERATORS:
                            for inp_idx in range(len(sibling.input)):
                                if sibling.input[inp_idx] == node.input[0]:
                                    self.replace_input.append(
                                        [sibling, sibling.input[inp_idx], self.model.get_children(node)[0].output[0]]
                                    )
            for node, old_input_name, new_input_name in self.replace_input:
                self.model.replace_node_input(node, old_input_name, new_input_name)
            self.model.update()

    def should_cast(self, node):
        """Check if node should be casted."""
        if node.name in self.config and self.config[node.name] != "fp32":  # pragma: no cover
            parent = self.model.get_parent(node, 0)
            if parent is not None and (parent.op_type != "Cast" or parent.attribute[0].i in [1, 10, 16]):
                return True
            elif parent is None and node.input[0] in self.model.input():
                return True
            return False

    def insert_qdq(self):
        """Insert Q/DQ pairs."""
        for node in self.model.nodes():
            if self.should_quantize(node):
                op_quantizer = OPERATORS[node.op_type](self, node)
                if op_quantizer.quantize_check():
                    op_quantizer.quantize()
            elif self.should_cast(node):  # pragma: no cover
                op_caster = OPERATORS[node.op_type](self, node)
                op_caster.cast()
        self.model.graph().node.extend(self.new_nodes)
        self.model.remove_nodes(self.remove_nodes)

        for node, old_input_name, new_input_name in self.replace_input:
            self.model.replace_node_input(node, old_input_name, new_input_name)
        self.model.update()

    def should_convert(self, node):
        """Check if node should be converted."""
        name = get_node_original_name(node)
        if (
            name in self.config
            and self.config[name] not in self.fallback_list
            and (self.config[name]["activation"]["quant_mode"] == "dynamic" or self.mode != "qdq")
        ):
            return True
        else:
            return False

    def convert_qdq_to_operator_oriented(self):
        """Convert QDQ to QOperator format."""
        self.new_nodes = []
        self.remove_nodes = []
        self.replace_input = []
        for node in self.model.nodes():
            if node.op_type not in ["QuantizeLinear", "DequantizeLinear"] and self.should_convert(node):
                op_converter = OPERATORS[node.op_type](self, node)
                mode = self.config[get_node_original_name(node)]["activation"]["quant_mode"]
                if op_converter.convert_check(mode):
                    op_converter.convert(mode)
        self.model.graph().node.extend(self.new_nodes)
        self.model.remove_nodes(self.remove_nodes)
        for node, old_input_name, new_input_name in self.replace_input:
            self.model.replace_node_input(node, old_input_name, new_input_name)
        self.model.update()

    def remove_redundant_pairs(self):
        """Remove redundant Q/DQ, Cast/Cast pairs."""
        self.remove_nodes = []
        self.replace_input = []
        pairs = [
            ["QuantizeLinear", "DequantizeLinear"],
            ["Cast", "Cast"],
        ]

        def dfs(match_nodes, node, pattern):
            if len(pattern) == 0:
                return
            start_id = 0
            end_id = len(pattern) - 1

            if node.op_type == pattern[end_id]:
                match_nodes.append(node)
            else:
                return

            if start_id == end_id:
                if all([i.op_type in ["QuantizeLinear", "DequantizeLinear"] for i in match_nodes]):
                    pair = [
                        str(find_by_name(i.input[2], self.model.initializer()).data_type) for i in match_nodes[::-1]
                    ]
                    if " ".join(pair) in support_pair and support_pair[" ".join(pair)]:
                        self.replace_input.append(
                            [
                                self.model.get_children(match_nodes[1])[0],
                                match_nodes[1].output[0],
                                match_nodes[0].input[0],
                            ]
                        )

                        self.remove_nodes.append(match_nodes[1])
                        if (
                            all(
                                [
                                    i.op_type in ["QuantizeLinear", "DequantizeLinear"]
                                    for i in self.model.get_children(match_nodes[0])
                                ]
                            )
                            and match_nodes[0].output[0] not in self.model.output()
                        ):
                            self.remove_nodes.append(match_nodes[0])
                else:  # pragma: no cover
                    parent = self.model.get_parents(match_nodes[0])[0]
                    children = self.model.get_children(match_nodes[1])
                    input_dtype = "1"  # float32
                    output_dtype = "1"  # 'float32'
                    outs = None
                    for inp in parent.input:
                        if inp in self.new_value_info:
                            input_dtype = str(self.new_value_info[inp].new_dtype)
                            break
                    for child in children:
                        outs = [out for out in child.output if out in self.new_value_info]
                        if len(outs) > 0:
                            output_dtype = str(self.new_value_info[outs[0]].new_dtype)
                            break
                    if outs is None or len(outs) == 0 or all([not self.should_cast(i) for i in children]):
                        return
                    if (
                        input_dtype == str(match_nodes[1].attribute[0].i)
                        and output_dtype == str(match_nodes[0].attribute[0].i)
                        and " ".join((output_dtype, input_dtype)) in support_pair
                        and support_pair[" ".join((output_dtype, input_dtype))]
                    ):
                        if match_nodes[0] not in self.remove_nodes and all(
                            [
                                i.op_type == "Cast" and str(i.attribute[0].i) == input_dtype
                                for i in self.model.get_children(match_nodes[0])
                            ]
                        ):
                            self.remove_nodes.append(match_nodes[0])
                        if match_nodes[1] not in self.remove_nodes:
                            self.remove_nodes.append(match_nodes[1])
                        for child in children:
                            self.replace_input.append(
                                [
                                    find_by_name(child.name, self.model.model.graph.node),
                                    match_nodes[1].output[0],
                                    match_nodes[0].input[0],
                                ]
                            )
                return

            children = self.model.get_children(node)
            for child in children:
                dfs(copy.deepcopy(match_nodes), child, pattern[:end_id])

        for n in self.model.nodes():
            matched = [i for i in pairs if n.op_type == i[-1]]
            if len(matched) > 0:
                for match_pair in matched:
                    visited_op = []
                    dfs(visited_op, n, match_pair)
        self.model.remove_nodes(self.remove_nodes)
        for node, old_input_name, new_input_name in self.replace_input:
            self.model.replace_node_input(node, old_input_name, new_input_name)
        self.model.update()

    def cast_inputs(self, node, cfg, indices=None):
        """Cast node input dtype."""
        for idx, tensor_name in enumerate(node.input):
            if indices and idx not in indices:
                continue
            initializer = find_by_name(tensor_name, self.model.initializer())
            if initializer is not None:
                if initializer.data_type != onnx_proto.TensorProto.FLOAT:
                    continue
                do_cast_new_tensor = cast_tensor(initializer, cfg, self.model.is_large_model)
                if do_cast_new_tensor:
                    # add cast initializer and update its name
                    self.model.add_initializer(do_cast_new_tensor)
                    node.input[idx] = do_cast_new_tensor.name

                    # if origin initializer is no more used, remove it
                    self.model.update()
                    input_name_to_nodes = self.model.input_name_to_nodes
                    if initializer.name not in input_name_to_nodes or len(input_name_to_nodes[initializer.name]) == 0:
                        self.model.remove_initializer(initializer)

                    self.new_value_info[do_cast_new_tensor.name] = ValueInfo(
                        do_cast_new_tensor.name, TensorProto.FLOAT, dtype_mapping[cfg]
                    )
            else:
                if (
                    tensor_name in self.value_infos
                    and self.value_infos[tensor_name].type.HasField("tensor_type")
                    and self.value_infos[tensor_name].type.tensor_type.elem_type != TensorProto.FLOAT
                ):
                    continue
                name = node.name + "_input_cast" + str(idx)
                self.new_nodes.append(
                    onnx.helper.make_node("Cast", [tensor_name], [name], to=dtype_mapping[cfg], name=name)
                )
                node.input[idx] = name
                self.new_value_info[name] = ValueInfo(tensor_name, TensorProto.FLOAT, dtype_mapping[cfg])

    def cast_outputs(self, node, cfg, indices=None):
        """Cast node output dtype."""
        for idx, tensor_name in enumerate(node.output):
            if indices and idx not in indices:
                continue
            if (
                tensor_name in self.value_infos
                and self.value_infos[tensor_name].type.HasField("tensor_type")
                and self.value_infos[tensor_name].type.tensor_type.elem_type != TensorProto.FLOAT
            ):
                continue
            node.output[idx] = tensor_name + "_to_cast_" + str(idx)
            name = node.name + "_output_cast" + str(idx)
            self.new_nodes.append(onnx.helper.make_node("Cast", [node.output[idx]], [tensor_name], to=1, name=name))
            self.new_value_info[node.output[idx]] = ValueInfo(tensor_name, dtype_mapping[cfg], TensorProto.FLOAT)

    def quantize_outputs(self, node, initializer_use_weight_qType=True, direct_int8=False):
        """Quantize node outputs."""
        if self.config[node.name]["activation"]["quant_mode"] == "dynamic":
            return
        for idx, tensor_name in enumerate(node.output):
            if (
                tensor_name in self.value_infos
                and self.value_infos[tensor_name].type.HasField("tensor_type")
                and self.value_infos[tensor_name].type.tensor_type.elem_type != TensorProto.FLOAT
            ):
                return
            data_found = False
            if direct_int8:
                if node.input[0] in self.quantized_value_map:
                    scale_name = self.quantized_value_map[node.input[0]].scale_name
                    zp_name = self.quantized_value_map[node.input[0]].zp_name
                    data_found = True
                elif node.input[0] in self.quantization_params:
                    data_found, scale_name, zp_name, _, _ = self._get_quantization_params(node.input[0])
            else:
                if tensor_name in self.quantized_value_map:
                    scale_name = self.quantized_value_map[tensor_name].scale_name
                    zp_name = self.quantized_value_map[tensor_name].zp_name
                    data_found = True
                elif tensor_name in self.quantization_params:
                    data_found, scale_name, zp_name, _, _ = self._get_quantization_params(tensor_name)

            if data_found is False:
                raise ValueError(
                    "Quantization parameters are not specified for param {}."
                    "In static mode quantization params for inputs and outputs "
                    "of nodes to be quantized are required.".format(tensor_name)
                )

            node.output[idx] = tensor_name + "_QuantizeInput"
            q_input = node.output[idx]
            q_output = tensor_name + "_quantized"
            dq_input = q_output
            dq_output = tensor_name
            quant_node_name = tensor_name + "_" + node.name + "_QuantizeLinear"
            dequant_node_name = tensor_name + "_" + node.name + "_DequantizeLinear"
            qlinear_node = make_quant_node(quant_node_name, [q_input, scale_name, zp_name], [q_output])
            dequant_node = make_dquant_node(dequant_node_name, [dq_input, scale_name, zp_name], [dq_output])
            self.new_nodes.extend([qlinear_node, dequant_node])
            quantized_value = QuantizedValue(tensor_name, dq_output, scale_name, zp_name, QuantizedValueType.Input)
            if tensor_name not in self.quantized_value_map:
                self.quantized_value_map[tensor_name] = quantized_value

    def quantize_inputs(self, node, indices=None, initializer_use_weight_qType=True, direct_int8=False):
        """Quantize node inputs."""
        # Quantize the input
        for idx, tensor_name in enumerate(node.input):
            if indices and idx not in indices:
                continue
            initializer = find_by_name(tensor_name, self.model.initializer())
            if initializer is not None:
                if initializer.data_type != onnx_proto.TensorProto.FLOAT:
                    return
                if node.op_type not in self.op_types_to_quantize:
                    dtype = (
                        onnx_proto.TensorProto.INT8 if initializer_use_weight_qType else onnx_proto.TensorProto.UINT8
                    )
                    scheme = "sym" if initializer_use_weight_qType else "asym"
                else:
                    dtype = (
                        self.config[node.name]["weight"]["dtype"]
                        if initializer_use_weight_qType
                        else self.config[node.name]["activation"]["dtype"]
                    )
                    scheme = (
                        self.config[node.name]["weight"]["scheme"]
                        if initializer_use_weight_qType
                        else self.config[node.name]["activation"]["scheme"]
                    )
                if self.add_qdq_pair_to_weight and self.mode == "qdq":
                    weight = self._get_quantized_weight(initializer, dtype, scheme)
                    self._update_weight(weight)
                    node.input[idx] = weight.name
                    q_weight_name = weight.name + "_quantized"
                    zp_name = weight.name + "_zero_point"
                    scale_name = weight.name + "_scale"
                    qlinear_node = make_quant_node(
                        weight.name + "_QuantizeLinear",
                        [tensor_name, scale_name, zp_name],
                        [weight.name + "_quantized"],
                    )
                    dequant_node = make_dquant_node(
                        weight.name + "_DequantizeLinear",
                        [weight.name + "_quantized", scale_name, zp_name],
                        [weight.name + "_dequantized"],
                    )
                    self.replace_input.append([node, weight.name, dequant_node.output[0]])
                    self.new_nodes.extend([qlinear_node, dequant_node])
                    quantized_value = QuantizedValue(
                        weight.name, q_weight_name, scale_name, zp_name, QuantizedValueType.Initializer, None, dtype
                    )
                    if weight.name not in self.quantized_value_map:
                        self.quantized_value_map[weight.name] = quantized_value
                else:
                    weight = self._get_quantized_weight(initializer, dtype, scheme)
                    self._update_weight(weight)
                    node.input[idx] = weight.name
                    q_weight_name = weight.name + "_quantized"
                    zp_name = weight.name + "_zero_point"
                    scale_name = weight.name + "_scale"

                    inputs = [q_weight_name, scale_name, zp_name]
                    output_name = tensor_name + "_DequantizeLinear"
                    dequant_node = onnx.helper.make_node(
                        "DequantizeLinear", inputs, [weight.name + "_dequantized"], weight.name + "_DequantizeLinear"
                    )
                    self.new_nodes.append(dequant_node)
                    self.replace_input.append([node, weight.name, dequant_node.output[0]])
                    quantized_value = QuantizedValue(
                        weight.name, q_weight_name, scale_name, zp_name, QuantizedValueType.Initializer, None, dtype
                    )
                    if weight.name not in self.quantized_value_map:
                        self.quantized_value_map[weight.name] = quantized_value
            else:
                if (
                    tensor_name in self.value_infos
                    and self.value_infos[tensor_name].type.HasField("tensor_type")
                    and self.value_infos[tensor_name].type.tensor_type.elem_type != TensorProto.FLOAT
                ):
                    return

                if tensor_name in self.quantized_value_map:
                    scale_name = self.quantized_value_map[tensor_name].scale_name
                    zp_name = self.quantized_value_map[tensor_name].zp_name
                    data_found = True
                else:
                    data_found, scale_name, zp_name, _, _ = self._get_quantization_params(tensor_name)
                if self.config[get_node_original_name(node)]["activation"]["quant_mode"] != "dynamic":
                    if data_found is False:
                        raise ValueError(
                            "Quantization parameters are not specified for param {}."
                            "In static mode quantization params for inputs and outputs "
                            "of nodes to be quantized are required.".format(tensor_name)
                        )
                    if direct_int8:
                        if node.input[0] not in self.quantized_value_map:
                            return
                    q_input = tensor_name
                    q_output = (
                        tensor_name + "_" + node.name + "_QuantizeLinear"
                        if tensor_name not in self.model.input()
                        else tensor_name + "_quantized"
                    )
                    dq_input = q_output
                    dq_output = (
                        tensor_name + "_" + node.name + "_dequantized"
                        if tensor_name not in self.model.input()
                        else tensor_name + "_dequantized"
                    )
                    self.replace_input.append([node, tensor_name, dq_output])
                    if tensor_name in self.model.input() and tensor_name in self.quantized_value_map:
                        continue

                    quant_node_name = tensor_name + "_" + node.name + "_QuantizeLinear"
                    dequant_node_name = tensor_name + "_" + node.name + "_DequantizeLinear"
                    qlinear_node = make_quant_node(quant_node_name, [q_input, scale_name, zp_name], [q_output])
                    dequant_node = make_dquant_node(dequant_node_name, [dq_input, scale_name, zp_name], [dq_output])
                    self.new_nodes.extend([qlinear_node, dequant_node])
                    quantized_value = QuantizedValue(
                        tensor_name, dq_output, scale_name, zp_name, QuantizedValueType.Input
                    )
                    if tensor_name not in self.quantized_value_map:
                        self.quantized_value_map[tensor_name] = quantized_value
                else:
                    qlinear_node = self.model.find_node_by_name(
                        tensor_name + "_QuantizeLinear", self.new_nodes, self.model.graph()
                    )
                    if qlinear_node is None:
                        if (
                            self.fuse_dynamic_quant
                            and self.config[node.name]["activation"]["dtype"] == onnx_proto.TensorProto.UINT8
                            and self.config[node.name]["activation"]["scheme"] == "asym"
                        ):
                            scale_name = tensor_name + "_scale"
                            zeropoint_name = tensor_name + "_zero_point"
                            if find_by_name(scale_name, self.model.initializer()):
                                self.model.remove_initializer(find_by_name(scale_name, self.model.initializer()))
                            if find_by_name(zeropoint_name, self.model.initializer()):
                                self.model.remove_initializer(find_by_name(zeropoint_name, self.model.initializer()))
                            qlinear_node = onnx.helper.make_node(
                                "DynamicQuantizeLinear",
                                [tensor_name],
                                [tensor_name + "_dynamic_quantized", scale_name, zeropoint_name],
                                tensor_name + "_QuantizeLinear",
                            )
                        else:
                            scale_name, zp_name, _, _ = self._get_dynamic_input_quantization_params(
                                tensor_name, self.config[node.name]["activation"]["dtype"]
                            )
                            qlinear_node = make_quant_node(
                                tensor_name + "_QuantizeLinear",
                                [tensor_name, scale_name, zp_name],
                                [tensor_name + "_quantized"],
                            )
                        if qlinear_node not in self.new_nodes:
                            self.new_nodes.append(qlinear_node)
                        self.quantized_value_map[tensor_name] = QuantizedValue(
                            tensor_name,
                            qlinear_node.output[0],
                            scale_name,
                            zp_name,
                            self.config[node.name]["activation"]["dtype"],
                        )
                    self.replace_input.append([node, tensor_name, qlinear_node.output[0]])

    def quantize_bias_tensor(self, node):
        """Quantize bias."""
        input_name, weight_name, bias_name = node.input
        if (
            self.quantization_params is None
            or input_name not in self.quantization_params
            or input_name not in self.quantized_value_map
            or (
                input_name in self.quantized_value_map
                and find_by_name(self.quantized_value_map[input_name].scale_name, self.model.initializer()) is None
            )
        ):
            self._dynamic_quantize_bias(input_name, weight_name + "_scale", bias_name, bias_name + "_quantized")
        else:
            beta = 1.0
            if node.op_type in ["Gemm"]:
                beta_attribute = [attr for attr in node.attribute if attr.name == "beta"]
                if len(beta_attribute):
                    beta = onnx.helper.get_attribute_value(beta_attribute[0])
            _, quant_value = self.quantize_bias(bias_name, input_name, weight_name, beta)
            if self.model.get_initializer_share_num(bias_name) == 1:
                self.model.remove_initializer(find_by_name(bias_name, self.model.initializer()))
            inputs = [quant_value.q_name, quant_value.scale_name, quant_value.zp_name]
            axis = None
            if find_by_name(weight_name + "_DequantizeLinear", self.new_nodes):
                dq_node = find_by_name(weight_name + "_DequantizeLinear", self.new_nodes)
                if dq_node.op_type == "DequantizeLinear" and find_by_name("axis", dq_node.attribute):
                    axis = find_by_name("axis", dq_node.attribute).i
            dequant_node = make_dquant_node(bias_name + "_DequantizeLinear", inputs, [bias_name + "_dequantized"], axis)
            self.new_nodes.append(dequant_node)
            self.replace_input.append(
                [find_by_name(node.name, self.model.nodes()), bias_name, bias_name + "_dequantized"]
            )

    def quantize_bias(self, bias_name, input_name, weight_name, beta=1.0):
        """Quantized the bias.

        Zero Point == 0 and Scale == Input_Scale * Weight_Scale
        """
        # get scale for weight
        weight_scale_initializer = find_by_name(weight_name + "_scale", self.model.initializer())
        weight_scale = (
            self.tensor_proto_to_array(weight_scale_initializer, os.path.dirname(self.model.model_path))
            if self.model.model_path is not None
            else self.tensor_proto_to_array(weight_scale_initializer)
        )

        # get bias
        bias_initializer = find_by_name(bias_name, self.model.initializer())
        bias_data = (
            self.tensor_proto_to_array(bias_initializer, os.path.dirname(self.model.model_path))
            if self.model.model_path is not None
            else self.tensor_proto_to_array(bias_initializer)
        )
        quantized_bias_name = bias_name + "_quantized"

        if input_name in self.quantized_value_map:
            input_scale_name = self.quantized_value_map[input_name].scale_name
        elif input_name in self.quantization_params:
            _, input_scale_name, _, _, _ = self._get_quantization_params(input_name)
        else:
            raise ValueError(f"Expected {input_name} to be in quantized value map for static quantization")
        inputscale_initializer = find_by_name(input_scale_name, self.model.initializer())
        input_scale = (
            self.tensor_proto_to_array(inputscale_initializer, os.path.dirname(self.model.model_path))
            if self.model.model_path is not None
            else self.tensor_proto_to_array(inputscale_initializer)
        )

        # calculate scale for bias

        bias_scale = input_scale * weight_scale * beta

        # quantize bias
        quantized_data = (np.asarray(bias_data) / bias_scale).round().astype(np.int32)

        # update bias initializer
        bias_np_data = np.asarray(quantized_data, dtype=np.int32).reshape(bias_initializer.dims)
        packed_bias_initializer = onnx.numpy_helper.from_array(bias_np_data, quantized_bias_name)
        self.model.initializer().extend([packed_bias_initializer])

        # update scale initializer
        quantized_bias_scale_name = bias_name + "_scale"
        bias_scale_data = np.asarray(bias_scale, dtype=np.float32).reshape(-1)
        packed_bias_scale_initializer = onnx.numpy_helper.from_array(bias_scale_data, quantized_bias_scale_name)
        self.model.initializer().extend([packed_bias_scale_initializer])

        # update zero initializer
        quantized_bias_zp_name = bias_name + "_zero_point"
        bias_zp_data = np.zeros(bias_scale.shape, dtype=np.int32).reshape(-1)
        packed_bias_zp_initializer = onnx.numpy_helper.from_array(bias_zp_data, quantized_bias_zp_name)
        self.model.initializer().extend([packed_bias_zp_initializer])

        # log entries for this quantized bias value
        quantized_bias_entry = QuantizedInitializer(
            bias_name,
            bias_initializer,
            [0],
            [0],
            [0],
            [bias_scale],
            bias_data,
            quantized_data,
            qType=onnx_proto.TensorProto.INT32,
        )

        quantized_value = QuantizedValue(
            bias_name,
            quantized_bias_name,
            quantized_bias_scale_name,
            quantized_bias_zp_name,
            QuantizedValueType.Initializer,
            None,
            onnx_proto.TensorProto.INT32,
        )
        return quantized_bias_name, quantized_value

    def _dynamic_quantize_bias(self, input_name, weight_scale_name, bias_name, quantized_bias_name):
        """Adds series of nodes required to quantize the bias dynamically.

        Args:
            input_name (string): Input name
            weight_scale_name (string): Weight scale name
            bias_name (string): bias name
            quantized_bias_name (string): quantized bias name
        """
        qType = onnx_proto.TensorProto.INT32
        input_scale_name = input_name + "_scale"
        bias_scale_node = onnx.helper.make_node(
            "Mul", [input_scale_name, weight_scale_name], [bias_name + "_scale"], bias_name + "_scale_node"
        )
        self.new_nodes.append(bias_scale_node)

        quantize_bias_node = onnx.helper.make_node(
            "Div", [bias_name, bias_scale_node.output[0]], [bias_name + "_tmp_quant:0"], bias_name + "_tmp_qaunt"
        )
        self.new_nodes.append(quantize_bias_node)

        bias_rounded_node = onnx.helper.make_node(
            "Floor", quantize_bias_node.output, [bias_name + "_quant_rounded:0"], bias_name + "_quant_rounded"
        )
        self.new_nodes.append(bias_rounded_node)

        bias_cast_node = onnx.helper.make_node(
            "Cast", bias_rounded_node.output, [quantized_bias_name], quantized_bias_name + "_node", to=qType
        )
        self.new_nodes.append(bias_cast_node)
        return

    def quantize_weights_per_channel(self, node, indices, weight_qType, scheme, axis):
        """Quantize weights per-channel."""
        if self.opset_version < 13 and self.mode == "qdq":
            self.quantize_inputs(node, indices)
            return

        for idx, inp in enumerate(node.input):
            if idx not in indices:
                continue

            if self.add_qdq_pair_to_weight and self.mode == "qdq":
                q_name, zp_name, scale_name = self.quantize_weight_per_channel(inp, weight_qType, scheme, axis)
                weight_name = (
                    ("_").join([inp, str(weight_qType)]) if self.model.get_initializer_share_num(inp) > 1 else inp
                )
                qlinear_node = make_quant_node(
                    weight_name + "_QuantizeLinear",
                    [inp, scale_name, zp_name],
                    [q_name],
                    axis,
                )
                dequant_node = make_dquant_node(
                    weight_name + "_DequantizeLinear",
                    [q_name, scale_name, zp_name],
                    [weight_name + "_dequantized"],
                    axis,
                )
                node.input[idx] = weight_name
                self.replace_input.append([node, weight_name, dequant_node.output[0]])
                self.new_nodes.extend([qlinear_node, dequant_node])
            else:
                q_name, zp_name, scale_name = self.quantize_weight_per_channel(inp, weight_qType, scheme, axis)
                weight_name = (
                    ("_").join([inp, str(weight_qType)]) if self.model.get_initializer_share_num(inp) > 1 else inp
                )
                dequant_node = make_dquant_node(
                    weight_name + "_DequantizeLinear",
                    [q_name, scale_name, zp_name],
                    [weight_name + "_dequantized"],
                    axis,
                )
                self.new_nodes.append(dequant_node)
                node.input[idx] = weight_name

                # Replace weight_name with output of DequantizeLinear
                self.replace_input.append([node, weight_name, dequant_node.output[0]])

    def quantize_weight_per_channel(self, weight_name, weight_qType, scheme, channel_axis):
        """Quantize weight per-channel."""
        name = (
            ("_").join([weight_name, str(weight_qType)])
            if self.model.get_initializer_share_num(weight_name) > 1
            else weight_name
        )
        if name in self.quantized_value_map:
            return (name + "_quantized", name + "_zero_point", name + "_scale")

        initializer = find_by_name(weight_name, self.model.initializer())
        if initializer is None:
            raise ValueError("{} is not an initializer", weight_name)

        weights = (
            self.tensor_proto_to_array(initializer, os.path.dirname(self.model.model_path))
            if self.model.model_path is not None
            else self.tensor_proto_to_array(initializer)
        )
        rmin, rmax, zero_point, scale, quantized_weights = quantize_data_per_channel(
            weights, channel_axis, _get_qrange_for_qType(weight_qType, self.reduce_range), weight_qType, scheme
        )

        weight = QuantizedInitializer(
            name,
            initializer,
            rmin,
            rmax,
            zero_point,
            scale,
            weights,
            quantized_weights.flatten().tolist(),
            channel_axis,
            weight_qType,
        )

        self._update_weight(weight)
        quantized_value = QuantizedValue(
            weight.name,
            weight.name + "_quantized",
            weight.name + "_scale",
            weight.name + "_zero_point",
            QuantizedValueType.Initializer,
            None,
            weight_qType,
        )
        self.quantized_value_map[weight.name] = quantized_value

        return (weight.name + "_quantized", weight.name + "_zero_point", weight.name + "_scale")

    def _update_weight(self, weight):
        """Update weight.

        Given a weight object, update the graph by doing the following:
         - remove old initializer, update new initializers for
           quantized weight, zero point, and scale
         - remove old weight input, update with new inputs for
           quantized weight, zero point, and scale
        This function does NOT update the nodes in the graph, just initializers and inputs
        """
        if weight.name in self.quantized_value_map:
            return
        packed_weight_name = weight.name + "_quantized"
        scale_name = weight.name + "_scale"
        zero_point_name = weight.name + "_zero_point"

        # Update packed weight, zero point, and scale initializers
        packed_weight_np_data = np.asarray(
            weight.quantized_data, dtype=onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[weight.qType]
        ).reshape(weight.initializer.dims)
        packed_weight_initializer = onnx.numpy_helper.from_array(packed_weight_np_data, packed_weight_name)

        if not self.add_qdq_pair_to_weight or self.mode != "qdq":
            self.model.initializer().append(packed_weight_initializer)
        if weight.axis is not None:
            zero_scale_shape = [weight.initializer.dims[weight.axis]]
        else:  # scale and zero point must be scalar
            zero_scale_shape = []
        zero_point_type = weight.qType
        scale_initializer = onnx.helper.make_tensor(
            scale_name, onnx_proto.TensorProto.FLOAT, zero_scale_shape, weight.scales
        )
        zero_initializer = onnx.helper.make_tensor(
            zero_point_name, zero_point_type, zero_scale_shape, weight.zero_points
        )

        self.model.initializer().extend([scale_initializer, zero_initializer])

    @staticmethod
    def tensor_proto_to_array(initializer, base_dir=""):
        """Convert TensorProto to array."""
        if initializer.data_type == onnx_proto.TensorProto.FLOAT:
            weights = onnx.numpy_helper.to_array(initializer, base_dir)
        else:
            raise ValueError(
                "Only float type quantization is supported. \
                Weights {} is {}.".format(
                    initializer.name, dtype_to_name(dtype_mapping, initializer.data_type)
                )
            )
        return weights

    def _get_quantization_params(self, param_name):
        """Create initializers and inputs in the graph for zero point and scale of output.

        Zero point and scale values are obtained from self.quantization_params if specified.

        Args:
            param_name (string): Name of the quantization parameter.
        """
        if self.quantization_params is None or param_name not in self.quantization_params:
            return False, "", "", "", ""

        params = self.quantization_params[param_name]
        if params is None or len(params) != 2:
            raise ValueError(
                "Quantization parameters should contain zero point and scale. "
                "Specified values for output {}: {}".format(param_name, params)
            )

        zero_point_values = [params[0]]
        zero_point_shape = []
        zero_point_name = param_name + "_zero_point"
        zero_point_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[params[0].dtype]

        scale_values = [params[1]]
        scale_shape = []
        scale_name = param_name + "_scale"

        # Add initializers
        init_zp = onnx.helper.make_tensor(zero_point_name, zero_point_type, zero_point_shape, zero_point_values)
        self.model.add_initializer(init_zp)
        init_scale = onnx.helper.make_tensor(scale_name, onnx_proto.TensorProto.FLOAT, scale_shape, scale_values)
        self.model.add_initializer(init_scale)

        return True, scale_name, zero_point_name, scale_shape, zero_point_shape

    def _get_quantized_weight(self, initializer, qType, scheme):
        """Get quantized weight."""
        name = (
            ("_").join([initializer.name, str(qType)])
            if self.model.get_initializer_share_num(initializer.name) > 1
            else initializer.name
        )
        if name in self.quantized_value_map:
            return self.quantized_value_map[name]
        weights_data = (
            self.tensor_proto_to_array(initializer, os.path.dirname(self.model.model_path))
            if self.model.model_path is not None
            else self.tensor_proto_to_array(initializer)
        )
        rmin, rmax, zero_point, scale, quantized_weights_data = quantize_data(
            weights_data.flatten().tolist(), _get_qrange_for_qType(qType, self.reduce_range), qType, scheme
        )
        weight = QuantizedInitializer(
            name,
            initializer,
            [rmin],
            [rmax],
            [zero_point],
            [scale],
            weights_data,
            quantized_weights_data,
            axis=None,
            qType=qType,
        )

        return weight

    def get_bias_add_nodes(self, node, weight_name, last_output, quantized_bias_name):
        """Given a node, this function handles bias add by adding a "reshape" node on bias and an "add" node.

        Args:
            node (NodeProto): current node (Conv)
            weight_name (string): weight name
            last_output (_type_): output of previous node (input to bias add)
            quantized_bias_name (string): bias name
        """
        # Add tensors for the shape to be reshaped to
        weight = find_by_name(weight_name, self.model.initializer())
        if weight is None:
            raise ValueError("Expected {} to be an initializer".format(node.input[1]))

        # Add reshape for correct broadcast
        reshape_input_data = quantized_bias_name
        reshape_input_shape = quantized_bias_name + "_reshape_shape"
        reshape_input = [reshape_input_data, reshape_input_shape]
        reshape_shape = np.ones((len(weight.dims)), dtype=np.int64)
        reshape_shape[1] = -1
        init_shape = onnx.helper.make_tensor(
            reshape_input_shape, onnx_proto.TensorProto.INT64, [len(weight.dims)], reshape_shape
        )
        self.model.add_initializer(init_shape)

        reshape_op_output = node.output[0] + "_reshape"
        reshape_node = onnx.helper.make_node(
            "Reshape", reshape_input, [reshape_op_output], quantized_bias_name + "reshape"
        )
        self.new_nodes.append(reshape_node)

        # Add an Add operation for bias
        bias_add_input = [last_output]
        bias_add_input.append(reshape_op_output)
        add_node_output = node.output[0] + "_bias_add"
        add_node = onnx.helper.make_node("Add", bias_add_input, [add_node_output], quantized_bias_name + "bias_add")
        self.new_nodes.append(add_node)
        return add_node_output

    def is_valid_quantize_weight(self, weight_name):
        """Check weight can be quantized."""
        weight = find_by_name(weight_name, self.model.initializer())
        if weight is not None:
            return weight.data_type == onnx_proto.TensorProto.FLOAT
        else:
            return weight_name in self.quantized_value_map

    def dequantize_tensor(self, node, value_name):
        """Dequantize tensor."""
        if value_name in self.quantized_value_map:
            quantized_value = self.quantized_value_map[value_name]
            dqlinear_name = value_name + "_DequantizeLinear"
            dqlinear_inputs = [value_name + "_quantized", quantized_value.scale_name, quantized_value.zp_name]
            dequantize_node = onnx.helper.make_node("DequantizeLinear", dqlinear_inputs, [value_name], dqlinear_name)
            if dequantize_node not in self.new_nodes:
                self.new_nodes.append(dequantize_node)
        else:  # pragma: no cover
            data_found, scale_name, zp_name, _, _ = self._get_quantization_params(value_name)
            if self.static:
                if data_found is False:
                    raise ValueError(
                        "Quantization parameters are not specified for param {}."
                        "In static mode quantization params for inputs and outputs "
                        "of nodes to be quantized are required.".format(value_name)
                    )
            dqlinear_name = value_name + "_DequantizeLinear"
            dqlinear_inputs = [value_name + "_quantized", scale_name, zp_name]
            dequantize_node = onnx.helper.make_node("DequantizeLinear", dqlinear_inputs, [value_name], dqlinear_name)
            if dequantize_node not in self.new_nodes:
                self.new_nodes.append(dequantize_node)

    def _get_dynamic_input_quantization_params(self, input_name, qType):
        """Create nodes for dynamic quantization of input.

        Args:
            input_name (string): Name of the input.
            qType (int): type to quantize to.
        """
        if qType == onnx_proto.TensorProto.INT8:
            return self._get_dynamic_input_quantization_params_int8(input_name)

        return self._get_dynamic_input_quantization_params_uint8(input_name)

    def _get_dynamic_input_quantization_params_int8(self, input_name):  # pragma: no cover
        """Create nodes for dynamic quantization of input to int8.

        Args:
            input_name (string): Name of the input.
        """
        qType = onnx_proto.TensorProto.INT8

        # Reduce min and Reduce max
        input_scale_name = input_name + "_scale"

        reduce_min_name = input_name + "_ReduceMin"
        reduce_min_node = onnx.helper.make_node(
            "ReduceMin",
            [input_name],
            [reduce_min_name + ":0"],
            reduce_min_name,
            keepdims=0,
        )
        self.new_nodes.append(reduce_min_node)

        reduce_max_name = input_name + "_ReduceMax"
        reduce_max_node = onnx.helper.make_node(
            "ReduceMax",
            [input_name],
            [reduce_max_name + ":0"],
            reduce_max_name,
            keepdims=0,
        )
        self.new_nodes.append(reduce_max_node)

        # Compute scale
        #   Find abs(rmin)
        reduce_min_abs_name = reduce_min_name + "_Abs"
        reduce_min_abs_node = onnx.helper.make_node(
            "Abs",
            [reduce_min_node.output[0]],
            [reduce_min_abs_name + ":0"],
            reduce_min_abs_name,
        )
        self.new_nodes.append(reduce_min_abs_node)
        #   Find abs(rmax)
        reduce_max_abs_name = reduce_max_name + "_Abs"
        reduce_max_abs_node = onnx.helper.make_node(
            "Abs",
            [reduce_max_node.output[0]],
            [reduce_max_abs_name + ":0"],
            reduce_max_abs_name,
        )
        self.new_nodes.append(reduce_max_abs_node)
        #   Compute max of abs(rmin) and abs(rmax)
        abs_max_name = input_name + "_Abs_Max"
        abs_max_node = onnx.helper.make_node(
            "Max",
            [reduce_min_abs_node.output[0], reduce_max_abs_node.output[0]],
            [abs_max_name + ":0"],
            abs_max_name,
        )
        self.new_nodes.append(abs_max_node)
        #   and divide by (quantize_range/2.0) which will be equal to max(...)*2.0/quantize_range
        initializer_div = onnx.helper.make_tensor(
            self.fixed_qrange_int8_name,
            onnx_proto.TensorProto.FLOAT,
            [],
            [_get_qrange_for_qType(qType) / 2.0],
        )
        self.model.add_initializer(initializer_div)
        scale_div_name = input_name + "scale_Div"
        scale_div_node = onnx.helper.make_node(
            "Div",
            [abs_max_node.output[0], self.fixed_qrange_int8_name],
            [input_scale_name],
            scale_div_name,
        )
        self.new_nodes.append(scale_div_node)

        # Zero point
        initializer_zp = onnx.helper.make_tensor(self.fixed_zero_zp_name, qType, [], [0])
        self.model.add_initializer(initializer_zp)

        return input_scale_name, self.fixed_zero_zp_name, [], []

    def _get_dynamic_input_quantization_params_uint8(self, input_name):
        """Create nodes for dynamic quantization of input to uint8.

        Args:
            input_name (string): Name of the input.
        """
        qType = onnx_proto.TensorProto.UINT8
        # Reduce min and Reduce max
        input_scale_name = input_name + "_scale"
        input_zp_name = input_name + "_zero_point"

        reduce_min_name = input_name + "_ReduceMin"
        reduce_min_node = onnx.helper.make_node(
            "ReduceMin",
            [input_name],
            [reduce_min_name + ":0"],
            reduce_min_name,
            keepdims=0,
        )
        self.new_nodes.append(reduce_min_node)

        reduce_max_name = input_name + "_ReduceMax"
        reduce_max_node = onnx.helper.make_node(
            "ReduceMax",
            [input_name],
            [reduce_max_name + ":0"],
            reduce_max_name,
            keepdims=0,
        )
        self.new_nodes.append(reduce_max_node)

        # Add tensors for quantize range and zero value.
        initializer_qrange = onnx.helper.make_tensor(
            self.fixed_qrange_uint8_name,
            onnx_proto.TensorProto.FLOAT,
            [],
            [_get_qrange_for_qType(qType)],
        )
        self.model.add_initializer(initializer_qrange)
        initializer_qvalue = onnx.helper.make_tensor(self.fixed_zero_name, onnx_proto.TensorProto.FLOAT, [], [0.0])
        self.model.add_initializer(initializer_qvalue)

        # Compute Scale
        #   Subtract rmax and rmin
        scale_sub_name = input_name + "_scale_Sub"
        scale_sub_node = onnx.helper.make_node(
            "Sub",
            [reduce_max_node.output[0], reduce_min_node.output[0]],
            [scale_sub_name + ":0"],
            scale_sub_name,
        )
        self.new_nodes.append(scale_sub_node)
        #   and divide by quantize range
        scale_div_name = input_name + "_scale_Div"
        scale_div_node = onnx.helper.make_node(
            "Div",
            [scale_sub_node.output[0], self.fixed_qrange_uint8_name],
            [input_scale_name],
            scale_div_name,
        )
        self.new_nodes.append(scale_div_node)

        # Compute zero point
        #   Subtract zero and rmin
        zp_sub_name = input_name + "_zero_point_Sub"
        zp_sub_node = onnx.helper.make_node(
            "Sub",
            [self.fixed_zero_name, reduce_min_node.output[0]],
            [zp_sub_name + ":0"],
            zp_sub_name,
        )
        self.new_nodes.append(zp_sub_node)
        #   Divide by scale
        zp_div_name = input_name + "_zero_point_Div"
        zp_div_node = onnx.helper.make_node(
            "Div",
            [zp_sub_node.output[0], input_scale_name],
            [zp_div_name + ":0"],
            zp_div_name,
        )
        self.new_nodes.append(zp_div_node)
        #   Compute floor
        zp_floor_name = input_name + "_zero_point_Floor"
        zp_floor_node = onnx.helper.make_node("Floor", zp_div_node.output, [zp_floor_name + ":0"], zp_floor_name)
        self.new_nodes.append(zp_floor_node)
        #   Cast to integer
        zp_cast_name = input_name + "_zero_point_Cast"
        zp_cast_node = onnx.helper.make_node("Cast", zp_floor_node.output, [input_zp_name], zp_cast_name, to=qType)
        self.new_nodes.append(zp_cast_node)

        return input_scale_name, input_zp_name, [], []
