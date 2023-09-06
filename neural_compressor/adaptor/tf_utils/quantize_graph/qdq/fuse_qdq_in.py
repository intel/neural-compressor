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
"""Quantize FusedInstanceNorm."""

from tensorflow.core.framework import graph_pb2, node_def_pb2
from tensorflow.python.framework import dtypes

from neural_compressor.adaptor.tf_utils.quantize_graph_common import QuantizeGraphHelper as helper

from ..quantize_graph_base import QuantizeNodeBase


class FuseNodeStartWithFusedInstanceNorm(QuantizeNodeBase):
    """Quantize FusedInstanceNorm and apply the fusion."""

    def __init__(self, **kwargs):
        """Initialization."""
        super().__init__(**kwargs)
        self.sorted_patterns = sorted(self.patterns, key=lambda i: len(i), reverse=True)
        if self.new_api:
            self.fusion_mapping = {
                "_MklFusedInstanceNormLeakyRelu": self.apply_newly_in_relu_fusion,
                "_MklFusedInstanceNormRelu": self.apply_newly_in_relu_fusion,
                "_MklFusedInstanceNorm": self.apply_newly_in_relu_fusion,
            }
        else:
            self.fusion_mapping = {}

    def apply_newly_in_relu_fusion(self, match_node_name):
        """Apply FusedInstanceNorm Relu/LeakyRelu fusion."""
        matched_node = self.node_name_mapping[match_node_name[0]]
        skip_node_name = match_node_name[1:]
        control_inputs, normal_inputs = self._get_node_input(matched_node.node.name)
        scale_name = normal_inputs[1]
        offset_name = normal_inputs[2]
        mean_name = normal_inputs[3]
        variance_name = normal_inputs[4]

        all_input_names = self._add_eightbit_prologue_nodes(matched_node.node.name)
        all_input_names = [
            all_input_names[0],
            scale_name,
            offset_name,
            mean_name,
            variance_name,
            all_input_names[1],
            all_input_names[2],
        ]

        for _, node in enumerate(self.input_graph.node):
            if node.name in skip_node_name:
                self.logger.debug("skip node {}".format(node.name))
            elif node.name == match_node_name[0]:
                self.logger.debug("Matched node {} with input {}.".format(node.name, node.input))

                relu_node_name = match_node_name[1] if len(match_node_name) == 2 else None

                node_op = "_QuantizedFusedInstanceNorm"
                quantized_node_name = node.name + "_eightbit_quantized_in"
                output_min_node_name = quantized_node_name + "_input7_output_min"
                output_max_node_name = quantized_node_name + "_input8_output_max"
                quantized_node_input_names = (
                    all_input_names + [output_min_node_name] + [output_max_node_name] + control_inputs
                )
                output_min_node = helper.create_constant_node(output_min_node_name, -1.0, dtypes.float32)
                output_max_node = helper.create_constant_node(output_max_node_name, 1.0, dtypes.float32)
                quantized_in_node = helper.create_node(node_op, quantized_node_name, quantized_node_input_names)

                if relu_node_name is not None:
                    relu_node = self.node_name_mapping[relu_node_name].node
                    if relu_node.op == "Relu":
                        helper.set_attr_string(quantized_in_node, "activation_mode", b"Relu")
                    elif relu_node.op == "LeakyRelu":
                        helper.set_attr_string(quantized_in_node, "activation_mode", b"LeakyRelu")
                        helper.set_attr_float(quantized_in_node, "leakyrelu_alpha", relu_node.attr["alpha"].f)

                helper.set_attr_dtype(quantized_in_node, "T", dtypes.qint8)
                helper.set_attr_dtype(quantized_in_node, "U", dtypes.float32)
                helper.set_attr_dtype(quantized_in_node, "Tout", dtypes.qint8)
                helper.copy_attr(quantized_in_node, "reduction_axes", node.attr["reduction_axes"])
                """# 0.

                x
                # 1. scale
                # 2. offset
                # 3. mean
                # 4. variance
                # 5. x_min
                # 6. x_max
                # 7. {output_min}
                # 8. {output_max}
                """
                helper.set_attr_type_list(
                    quantized_in_node,
                    "input_types",
                    [
                        dtypes.qint8.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ],
                )
                """# 0.

                output
                # 1. output_min
                # 2. output_max
                """
                helper.set_attr_type_list(
                    quantized_in_node,
                    "out_types",
                    [
                        dtypes.qint8.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ],
                )
                self.add_output_graph_node(output_min_node)
                self.add_output_graph_node(output_max_node)
                self.add_output_graph_node(quantized_in_node)
                self._intel_cpu_add_dequantize_result_node(
                    quantized_output_name=quantized_node_name,
                    original_node_name=match_node_name[-1],
                    dtype=dtypes.qint8,
                    min_tensor_index=1,
                )

            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)

    def get_longest_fuse(self):
        """Get the longest fusion pattern."""
        self._get_op_list()
        real_patterns = [pattern[1:-1] for pattern in self.sorted_patterns]
        # Cannot match if: self._is_match([['Q','IN','LeakyRelu','DQ'],['Q','IN','Relu','DQ'],['Q','IN','DQ']])
        matched_rule, matched_node_name = self._is_match(real_patterns)
        return matched_rule, matched_node_name

    def apply_the_transform(self):
        """Quantize FusedInstanceNorm and apply the fusion pattern."""
        self._get_op_list()
        real_patterns = [pattern[1:-1] for pattern in self.sorted_patterns]
        # Cannot match if: self._is_match([['Q','IN','LeakyRelu','DQ'],['Q','IN','Relu','DQ'],['Q','IN','DQ']])
        matched_rule, matched_node_name = self._is_match(real_patterns)
        if matched_node_name:
            self.output_graph = graph_pb2.GraphDef()
            fusion_name = "".join(matched_rule)
            if fusion_name in self.fusion_mapping:
                self.fusion_mapping[fusion_name](matched_node_name)
            else:
                if self.new_api:
                    self.logger.info("Unknown fusion pattern {} .".format(fusion_name))
                if self.remove_redundant_quant_flag:
                    self.input_graph = self.remove_redundant_quantization(self.input_graph)
                return self.input_graph, []

            self.input_graph = self.output_graph
            self._reset_output_node_maps()
            if self.remove_redundant_quant_flag:
                self.output_graph = self.remove_redundant_quantization(self.output_graph)
            return self.output_graph, []
        return self.input_graph, []
