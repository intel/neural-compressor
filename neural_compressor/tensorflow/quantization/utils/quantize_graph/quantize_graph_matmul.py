#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
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
"""Quantize MatMul."""

import numpy as np
from tensorflow.core.framework import graph_pb2, node_def_pb2
from tensorflow.python.framework import dtypes, tensor_util

from neural_compressor.tensorflow.quantization.utils.quantize_graph_common import QuantizeGraphHelper as helper

from .quantize_graph_base import QuantizeNodeBase


class FuseNodeStartWithMatmul(QuantizeNodeBase):
    """Quantize MatMul and apply the fusion."""

    def __init__(self, **kwargs):
        """Initialization."""
        super().__init__(**kwargs)

        self.sorted_patterns = sorted(self.patterns, key=lambda i: len(i), reverse=True)
        self.exclude_matmul_name = []
        self.fusion_op_type = set(fusion[0] for fusion in self.patterns)
        self.fusion_mapping = {
            "MatMulBiasAdd": self.apply_matmul_biasadd_fusion,
            "MatMul": self.apply_matmul_biasadd_fusion,
            "MatMulBiasAddRelu": self.apply_matmul_biasadd_relu_fusion,
            "MatMulRelu": self.apply_matmul_biasadd_relu_fusion,
        }

    def apply_matmul_biasadd_relu_fusion(self, match_node_name):
        """Apply the MatMul BiasAdd Relu fusion."""
        matched_node = self.node_name_mapping[match_node_name[0]]
        control_inputs, normal_inputs = self._get_node_input(matched_node.node.name)
        weight_name = normal_inputs[1]
        weight_node = self.node_name_mapping[helper.node_name_from_input(weight_name)].node

        # FIXME We only quantize the MatMul op which second input node type is const. This is a
        # workaround for RNN model like LTSM.
        if weight_node.op != "Const":
            self.output_graph = self.input_graph
            return []

        weights_content = tensor_util.MakeNdarray(weight_node.attr["value"].tensor)

        if np.any(np.isnan(weights_content)):
            self.output_graph = self.input_graph
            return []

        for i in self.node_name_mapping:
            if weight_node.name in self.node_name_mapping[i].output:
                self.output_graph = self.input_graph
                return []
        second_node = self.node_name_mapping[match_node_name[1]].node
        skip_node_name = match_node_name[1:]

        need_insert_dummy_biasadd = 1
        offset = 1
        if len(match_node_name) == 3:
            add_a_node_name = helper.node_name_from_input(second_node.input[0])
            add_a_node = self.node_name_mapping[add_a_node_name].node
            add_b_node_name = helper.node_name_from_input(second_node.input[1])
            add_b_node = self.node_name_mapping[add_b_node_name].node
            if add_a_node.op != "Const" and add_b_node.op == "Const":
                need_insert_dummy_biasadd = 0
                offset = 0
            if need_insert_dummy_biasadd:
                self.apply_matmul_biasadd_fusion(match_node_name[:1])
                return match_node_name[:1]

        q_weights_name, q_weights_min_name, q_weights_max_name = self._intel_cpu_quantize_weight_eightbit(
            matched_node.node.op, self.node_name_mapping[weight_name].node, self.per_channel
        )

        skip_node_name.append(weight_name)

        for _, node in enumerate(self.input_graph.node):
            if node.name in skip_node_name:
                pass
            elif node.name == match_node_name[0]:
                self.logger.debug("Matched node {} with input {}.".format(node.name, node.input))

                quantized_node_name = node.name + "_eightbit_quantized_mat_mul"
                if need_insert_dummy_biasadd:
                    t_b_index = 0 if matched_node.node.attr["transpose_b"].b else 1
                    bias_size = weights_content.shape[t_b_index]
                    bias_node_name = node.name + "_fake_bias"
                    bias_node = helper.create_constant_node(
                        bias_node_name, [0] * bias_size, dtypes.float32, shape=[bias_size]
                    )
                    self.add_output_graph_node(bias_node)
                else:
                    bias_node_name = self.node_name_mapping[match_node_name[1]].node.input[1]

                relu_node_name = match_node_name[2 - offset]
                all_input_names = self._add_eightbit_prologue_nodes(matched_node.node.name)
                all_input_names = all_input_names[:1] + [q_weights_name] + all_input_names[1:]
                all_input_names.append(q_weights_min_name)
                all_input_names.append(q_weights_max_name)
                quantized_node_input_names = (
                    all_input_names[:2] + [bias_node_name] + all_input_names[2:] + control_inputs
                )

                quantized_matmul_node = helper.create_node(
                    "QuantizedMatMulWithBiasAndRelu", quantized_node_name, quantized_node_input_names
                )

                helper.copy_attr(quantized_matmul_node, "transpose_a", node.attr["transpose_a"])
                helper.copy_attr(quantized_matmul_node, "transpose_b", node.attr["transpose_b"])
                helper.set_attr_dtype(quantized_matmul_node, "T1", dtypes.quint8)
                helper.set_attr_dtype(quantized_matmul_node, "T2", dtypes.qint8)
                helper.set_attr_dtype(quantized_matmul_node, "Toutput", dtypes.qint32)
                helper.set_attr_string(
                    quantized_matmul_node, "input_quant_mode", b"MIN_FIRST" if self.is_asymmetric else b"SCALED"
                )

                self.add_output_graph_node(quantized_matmul_node)

                quantize_down_name = self._add_quantize_down_nodes(node, quantized_node_name, dtypes.quint8, False)
                self._intel_cpu_add_dequantize_result_node(quantize_down_name, relu_node_name)
            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)
        return match_node_name

    def apply_matmul_biasadd_fusion(self, match_node_name):
        """Apply MatMul BiasAdd fusion."""
        skip_node_name = match_node_name[1:]
        matched_node = self.node_name_mapping[match_node_name[0]]
        control_inputs, normal_inputs = self._get_node_input(matched_node.node.name)
        weight_name = normal_inputs[1]
        weight_node = self.node_name_mapping[helper.node_name_from_input(weight_name)].node

        enter_node = None

        if weight_node.op == "Enter":
            parent_node = self.node_name_mapping[helper.node_name_from_input(weight_node.input[0])].node
            # FIXME We only quantize the MatMul op which second input node type is const. This is a
            # workaround for RNN model like LTSM.
            if parent_node.op != "Const":
                self.logger.debug("The weight node of matched_node {} is not Const or Const + Enter, skipped")
                self.output_graph = self.input_graph
                return []
            enter_node = weight_node
            weight_node = parent_node
            weight_name = weight_node.name

        if weight_node.op != "Const":
            self.output_graph = self.input_graph
            return []

        # TODO Remove below two lines once the TF enabled the old QuantizedMatMul while
        # transpose_a/transpose_b could be set to True.
        if matched_node.node.attr["transpose_a"].b is True or matched_node.node.attr["transpose_b"].b is True:
            self.exclude_matmul_name.append(match_node_name[0])
            self.output_graph = self.input_graph
            return []

        if weight_node.op == "Const":
            weights_content = tensor_util.MakeNdarray(weight_node.attr["value"].tensor)

            if np.any(np.isnan(weights_content)):
                self.output_graph = self.input_graph
                return []

        for i in self.node_name_mapping:
            if (
                weight_node.input
                and not weight_node.input[0].startswith("^")
                and weight_node.name in self.node_name_mapping[i].output
            ):
                self.output_graph = self.input_graph
                return []

        len_output = len(matched_node.output)
        is_shared_output = False
        if len_output == 2:
            if (
                self.node_name_mapping[matched_node.output[0]].node.op == "Reshape"
                or self.node_name_mapping[matched_node.output[1]].node.op == "Reshape"
            ):
                is_shared_output = False
            else:
                is_shared_output = True
        elif len_output > 1:
            is_shared_output = True

        need_insert_dummy_biasadd = 1
        if len(match_node_name) == 1:
            if is_shared_output:
                self.output_graph = self.input_graph
                return []
        else:
            second_node = self.node_name_mapping[match_node_name[1]].node
            add_a_node_name = helper.node_name_from_input(second_node.input[0])
            add_a_node = self.node_name_mapping[add_a_node_name].node
            add_b_node_name = helper.node_name_from_input(second_node.input[1])
            add_b_node = self.node_name_mapping[add_b_node_name].node
            if add_a_node.op != "Const" and add_b_node.op in ("Const", "Enter"):
                need_insert_dummy_biasadd = 0
            if need_insert_dummy_biasadd:
                self.apply_matmul_biasadd_fusion(match_node_name[:1])
                return match_node_name[:1]

        if self.frame_info and not enter_node:
            from collections import OrderedDict

            frame_info = OrderedDict(self.frame_info)
            if match_node_name[0] in frame_info and frame_info[match_node_name[0]]:
                enter_node = helper.create_node("Enter", weight_name + "_enter", [weight_name])
                helper.set_attr_string(enter_node, "frame_name", frame_info[weight_name].attr["frame_name"].s)
                helper.set_attr_dtype(enter_node, "T", dtypes.float32)
                helper.set_attr_bool(enter_node, "is_constant", True)
                helper.set_attr_int(
                    enter_node, "parallel_iterations", frame_info[weight_name].attr["parallel_iterations"].i
                )

        q_weights_name, q_weights_min_name, q_weights_max_name = self._intel_cpu_quantize_weight_eightbit(
            matched_node.node.op, self.node_name_mapping[weight_name].node, self.per_channel, enter_node
        )

        skip_node_name.append(weight_name)
        if enter_node:
            skip_node_name.append(enter_node.name)

        for _, node in enumerate(self.input_graph.node):
            if node.name in skip_node_name:
                pass
            elif node.name == match_node_name[0]:
                self.logger.debug("Matched node {} with input {}.".format(node.name, node.input))

                quantized_node_name = node.name + "_eightbit_quantized_mat_mul"

                if need_insert_dummy_biasadd:
                    t_b_index = 0 if matched_node.node.attr["transpose_b"].b else 1
                    bias_size = weights_content.shape[t_b_index]
                    bias_node_name = node.name + "_fake_bias"
                    bias_node = helper.create_constant_node(
                        bias_node_name, [0] * bias_size, dtypes.float32, shape=[bias_size]
                    )
                    if enter_node:
                        bias_enter_node = helper.create_node("Enter", bias_node_name + "_enter", [bias_node_name])
                        helper.set_attr_string(bias_enter_node, "frame_name", enter_node.attr["frame_name"].s)
                        helper.set_attr_dtype(bias_enter_node, "T", dtypes.float32)
                        helper.set_attr_bool(bias_enter_node, "is_constant", True)
                        helper.set_attr_int(
                            bias_enter_node, "parallel_iterations", enter_node.attr["parallel_iterations"].i
                        )

                        self.add_output_graph_node(bias_enter_node)
                        bias_node_name = bias_enter_node.name

                    self.add_output_graph_node(bias_node)
                else:
                    bias_node_name = self.node_name_mapping[match_node_name[1]].node.input[1]
                    if self.node_name_mapping[bias_node_name].node.op == "Enter":
                        bias_enter_node = helper.create_node("Enter", bias_node_name + "_enter", [bias_node_name])
                        helper.set_attr_string(
                            bias_enter_node,
                            "frame_name",
                            self.node_name_mapping[bias_node_name].node.attr["frame_name"].s,
                        )
                        helper.set_attr_dtype(bias_enter_node, "T", dtypes.float32)
                        helper.set_attr_bool(bias_enter_node, "is_constant", True)
                        helper.set_attr_int(
                            bias_enter_node,
                            "parallel_iterations",
                            self.node_name_mapping[bias_node_name].node.attr["parallel_iterations"].i,
                        )
                        self.add_output_graph_node(bias_enter_node)

                all_input_names = self._add_eightbit_prologue_nodes(matched_node.node.name)
                all_input_names = all_input_names[:1] + [q_weights_name] + all_input_names[1:]
                all_input_names.append(q_weights_min_name)
                all_input_names.append(q_weights_max_name)
                quantized_node_input_names = (
                    all_input_names[:2] + [bias_node_name] + all_input_names[2:] + control_inputs
                )

                quantized_matmul_node = helper.create_node(
                    "QuantizedMatMulWithBias", quantized_node_name, quantized_node_input_names
                )

                helper.copy_attr(quantized_matmul_node, "transpose_a", node.attr["transpose_a"])
                helper.copy_attr(quantized_matmul_node, "transpose_b", node.attr["transpose_b"])
                helper.set_attr_dtype(quantized_matmul_node, "T1", dtypes.quint8)
                helper.set_attr_dtype(quantized_matmul_node, "T2", dtypes.qint8)
                helper.set_attr_dtype(quantized_matmul_node, "Toutput", dtypes.qint32)
                helper.set_attr_dtype(quantized_matmul_node, "Tbias", dtypes.float32)
                helper.set_attr_string(
                    quantized_matmul_node, "input_quant_mode", b"MIN_FIRST" if self.is_asymmetric else b"SCALED"
                )

                self.add_output_graph_node(quantized_matmul_node)
                requantize_type = dtypes.qint8

                quantize_down_name = self._add_quantize_down_nodes(node, quantized_node_name, requantize_type, False)
                self._intel_cpu_add_dequantize_result_node(
                    quantize_down_name,
                    match_node_name[0] if need_insert_dummy_biasadd else match_node_name[1],
                    requantize_type,
                )
            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)
        return match_node_name

    def get_longest_fuse(self):
        """Get the longest fusion pattern."""
        self._get_op_list()
        matched_rule, _ = self._is_match(self.sorted_patterns)
        return matched_rule

    def apply_the_transform(self):
        """Quantize MatMul and apply the fusion pattern."""
        self._get_op_list()
        matched_rule, matched_node_name = self._is_match(self.sorted_patterns)

        if matched_node_name:
            self.output_graph = graph_pb2.GraphDef()
            fusion_name = "".join(matched_rule)
            if fusion_name in self.fusion_mapping:
                matched_nodes = self.fusion_mapping[fusion_name](matched_node_name)
            else:  # pragma: no cover
                self.logger.debug("Unknown fusion pattern {}.".format(fusion_name))
                if self.remove_redundant_quant_flag:
                    self.input_graph = self.remove_redundant_quantization(self.input_graph)
                return self.input_graph, []
            self.input_graph = self.output_graph
            self._reset_output_node_maps()
            if self.remove_redundant_quant_flag:
                self.output_graph = self.remove_redundant_quantization(self.output_graph)
            return self.output_graph, matched_nodes, self.exclude_matmul_name

        return self.input_graph, [], []
