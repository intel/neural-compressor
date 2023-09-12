#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
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
"""Quantize the AvgPool/MaxPool."""

import tensorflow as tf
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import dtypes

from neural_compressor.adaptor.tf_utils.quantize_graph_common import QuantizeGraphHelper as helper
from neural_compressor.adaptor.tf_utils.util import version1_eq_version2, version1_gt_version2, version1_lt_version2

from ..quantize_graph_base import QuantizeNodeBase


class FuseNodeStartWithPooling(QuantizeNodeBase):
    """Quantize the AvgPool and MaxPool."""

    def __init__(self, **kwargs):
        """Initialization."""
        super().__init__(**kwargs)
        self.sorted_patterns = sorted(self.patterns, key=lambda i: len(i), reverse=True)

    def _add_pool_function(self, original_node, quantized_op_node):
        """Set quantized pooling node attributes."""
        pooling_type = (
            dtypes.quint8
            if version1_lt_version2(tf.version.VERSION, "2.6.0") or self._find_relu_node(original_node)
            else dtypes.qint8
        )
        helper.set_attr_dtype(quantized_op_node, "T", pooling_type)
        helper.copy_attr(quantized_op_node, "ksize", original_node.attr["ksize"])
        helper.copy_attr(quantized_op_node, "strides", original_node.attr["strides"])
        helper.copy_attr(quantized_op_node, "padding", original_node.attr["padding"])

    def _apply_pool_quantization(self, match_node_name):
        """Quantize AvgPool/MaxPool.

        Dequantize + MaxPool + QuantizeV2
        Dequantize + MaxPool3D + QuantizeV2
        Dequantize + AvgPool + QuantizeV2
        """
        skip_node_name = match_node_name[2:]
        matched_node = self.node_name_mapping[match_node_name[1]]
        control_inputs, normal_inputs = self._get_node_input(matched_node.node.name)
        _, q_inputs = self._get_node_input(normal_inputs[0])

        all_input_names = q_inputs
        skip_node_name.append(normal_inputs[0])

        for _, node in enumerate(self.input_graph.node):
            if node.name in skip_node_name:
                self.logger.debug("skip node {}".format(node.name))
            elif node.name == match_node_name[1]:
                self.logger.debug("Matched node {} with input {}.".format(node.name, node.input))
                quantized_op_name = node.name + "_eightbit_quantized"
                quantized_op_type = "Quantized" + node.op
                if node.op == "MaxPool3D":
                    quantized_op_type = "_Quantized" + node.op

                quantized_pool_node = helper.create_node(quantized_op_type, quantized_op_name, all_input_names)

                self._add_pool_function(node, quantized_pool_node)
                self.add_output_graph_node(quantized_pool_node)
                deq_type = dtypes.quint8 if self._find_relu_node(node) else dtypes.qint8
                self._intel_cpu_add_dequantize_result_node(
                    quantized_op_name, node.name, dtype=deq_type, performance_only=self.performance_only
                )
            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)

    def get_longest_fuse(self):
        """Get the longest fusion pattern."""
        self._get_op_list()
        matched_node_name = []

        for k, v in enumerate(self.op_list):
            if v in set(fusion[1] for fusion in self.sorted_patterns):
                cur_node = self.node_name_mapping[list(self.node_name_mapping.keys())[k]].node

                if cur_node.name != self.start_node_name:
                    continue

                for sub_rule in self.sorted_patterns:
                    if sub_rule[0] != "Dequantize" or sub_rule[-1] != "QuantizeV2":
                        continue
                    if v != sub_rule[1]:
                        continue
                    matched_node_name.clear()
                    matched_node_name.append(sub_rule[0])
                    matched_node_name.append(cur_node.name)
                    matched_node_name.append(sub_rule[-1])
                    return sub_rule, matched_node_name
        return None, None

    def apply_the_transform(self):
        """Quantize AvgPool/MaxPool."""
        self._get_op_list()
        matched_rule, matched_node_name = self.get_longest_fuse()
        if matched_node_name:
            fusion_name = "".join(matched_rule)
            if (
                fusion_name == "DequantizeMaxPoolQuantizeV2"
                or fusion_name == "DequantizeMaxPool3DQuantizeV2"
                or fusion_name == "DequantizeAvgPoolQuantizeV2"
            ):
                self._apply_pool_quantization(matched_node_name)
            else:  # pragma: no cover
                self.logger.info("Unknown fusion pattern {}.".format(fusion_name))
                if self.remove_redundant_quant_flag:
                    self.input_graph = self.remove_redundant_quantization(self.input_graph)
                return self.input_graph, []

            self.input_graph = self.output_graph
            self._reset_output_node_maps()
            if self.remove_redundant_quant_flag:
                self.output_graph = self.remove_redundant_quantization(self.output_graph)
            return self.output_graph, []

        if self.remove_redundant_quant_flag:
            self.input_graph = self.remove_redundant_quantization(self.input_graph)
        return self.input_graph, []
