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
"""Quantize FusedBatchNormV3 to int8 op."""

from tensorflow.core.framework import graph_pb2, node_def_pb2
from tensorflow.python.framework import dtypes

from neural_compressor.tensorflow.quantization.utils.quantize_graph_common import QuantizeGraphHelper as helper

from ..quantize_graph_base import QuantizeNodeBase


class FuseNodeStartWithFusedBatchNormV3(QuantizeNodeBase):
    """Quantize FusedBatchNormV3 to int8 op _QuantizedFusedBatchNorm."""

    def __init__(self, **kwargs):
        """Initialization."""
        super().__init__(**kwargs)
        self.sorted_patterns = sorted(self.patterns, key=lambda i: len(i), reverse=True)
        if self.new_api:
            self.fusion_mapping = {
                "FusedBatchNormV3": self.apply_newly_bn_relu_fusion,
                "FusedBatchNormV3Relu": self.apply_newly_bn_relu_fusion,
                "FusedBatchNormV3LeakyRelu": self.apply_newly_bn_leakyrelu_fusion,
            }
        else:
            self.fusion_mapping = {}
        self.exclude_bn_nodes = []

    def apply_newly_bn_relu_fusion(self, match_node_name):
        """Apply the BN + Relu fusion."""
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

                node_op = "_QuantizedFusedBatchNorm"
                quantized_node_name = node.name + "_eightbit_quantized_bn"
                output_min_node_name = quantized_node_name + "_input7_output_min"
                output_max_node_name = quantized_node_name + "_input8_output_max"
                quantized_node_input_names = (
                    all_input_names + [output_min_node_name] + [output_max_node_name] + control_inputs
                )
                output_min_node = helper.create_constant_node(output_min_node_name, -1.0, dtypes.float32)
                output_max_node = helper.create_constant_node(output_max_node_name, 1.0, dtypes.float32)
                quantized_bn_node = helper.create_node(node_op, quantized_node_name, quantized_node_input_names)
                if relu_node_name is not None:
                    helper.set_attr_string(quantized_bn_node, "activation_mode", b"Relu")
                if self.node_name_mapping[offset_name].node.op == "Const":
                    helper.set_attr_bool(quantized_bn_node, "is_offset_const", True)
                else:
                    helper.set_attr_bool(quantized_bn_node, "is_offset_const", False)
                if self.node_name_mapping[mean_name].node.op == "Const":
                    helper.set_attr_bool(quantized_bn_node, "is_mean_const", True)
                else:
                    helper.set_attr_bool(quantized_bn_node, "is_mean_const", False)
                helper.set_attr_dtype(quantized_bn_node, "T", dtypes.qint8)
                helper.set_attr_dtype(quantized_bn_node, "U", dtypes.float32)
                helper.set_attr_dtype(quantized_bn_node, "Tout", dtypes.qint8)
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
                    quantized_bn_node,
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
                    quantized_bn_node,
                    "out_types",
                    [
                        dtypes.qint8.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ],
                )
                self.add_output_graph_node(output_min_node)
                self.add_output_graph_node(output_max_node)
                self.add_output_graph_node(quantized_bn_node)
                self._intel_cpu_add_dequantize_result_node(
                    quantized_output_name=quantized_node_name,
                    original_node_name=match_node_name[-1],
                    dtype=dtypes.qint8,
                    min_tensor_index=1,
                    performance_only=self.performance_only,
                )

            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)

    def apply_newly_bn_leakyrelu_fusion(self, match_node_name):
        """Apply BN + LeakyRelu fusion."""
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
                leakyrelu_node_name = match_node_name[1]
                node_op = "_QuantizedFusedBatchNorm"
                quantized_node_name = node.name + "_eightbit_quantized_bn"
                output_min_node_name = quantized_node_name + "_input7_output_min"
                output_max_node_name = quantized_node_name + "_input8_output_max"
                quantized_node_input_names = (
                    all_input_names + [output_min_node_name] + [output_max_node_name] + control_inputs
                )
                output_min_node = helper.create_constant_node(output_min_node_name, -1.0, dtypes.float32)
                output_max_node = helper.create_constant_node(output_max_node_name, 1.0, dtypes.float32)
                quantized_bn_node = helper.create_node(node_op, quantized_node_name, quantized_node_input_names)

                helper.set_attr_string(quantized_bn_node, "activation_mode", b"LeakyRelu")
                helper.copy_attr(
                    quantized_bn_node, "alpha", self.node_name_mapping[leakyrelu_node_name].node.attr["alpha"]
                )
                if self.node_name_mapping[offset_name].node.op == "Const":
                    helper.set_attr_bool(quantized_bn_node, "is_offset_const", True)
                else:
                    helper.set_attr_bool(quantized_bn_node, "is_offset_const", False)
                if self.node_name_mapping[mean_name].node.op == "Const":
                    helper.set_attr_bool(quantized_bn_node, "is_mean_const", True)
                else:
                    helper.set_attr_bool(quantized_bn_node, "is_mean_const", False)
                helper.set_attr_dtype(quantized_bn_node, "T", dtypes.qint8)
                helper.set_attr_dtype(quantized_bn_node, "U", dtypes.float32)
                helper.set_attr_dtype(quantized_bn_node, "Tout", dtypes.qint8)
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
                    quantized_bn_node,
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
                    quantized_bn_node,
                    "out_types",
                    [
                        dtypes.qint8.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ],
                )
                self.add_output_graph_node(output_min_node)
                self.add_output_graph_node(output_max_node)
                self.add_output_graph_node(quantized_bn_node)
                self._intel_cpu_add_dequantize_result_node(
                    quantized_output_name=quantized_node_name,
                    original_node_name=match_node_name[-1],
                    dtype=dtypes.qint8,
                    min_tensor_index=1,
                    performance_only=self.performance_only,
                )

            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)

    def get_longest_fuse(self):
        """Get the longest fusion pattern."""
        self._get_op_list()
        real_patterns = [pattern[1:-1] for pattern in self.sorted_patterns]
        # Cannot match if: self._is_match([['Q','BN','Relu','DQ']],['Q','BN','DQ']])
        matched_rule, matched_node_name = self._is_match(real_patterns)
        return matched_rule, matched_node_name

    def apply_the_transform(self):
        """Apply the BN int8 fusion."""
        self._get_op_list()
        real_patterns = [pattern[1:-1] for pattern in self.sorted_patterns]
        # Cannot match if: self._is_match([['Q','BN','Relu','DQ']],['Q','BN','DQ']])
        matched_rule, matched_node_name = self._is_match(real_patterns)
        if matched_node_name:
            self.output_graph = graph_pb2.GraphDef()
            fusion_name = "".join(matched_rule)
            bn_node = self.node_name_mapping[matched_node_name[0]].node
            is_training = bn_node.attr["is_training"].b
            if fusion_name in self.fusion_mapping and is_training is False:
                self.fusion_mapping[fusion_name](matched_node_name)
            else:
                if is_training is True:
                    self.logger.info(
                        "Skip quantizing the BN node '{}' due to the attr 'is_training == true'.".format(bn_node.name)
                    )
                    self.exclude_bn_nodes.append(bn_node.name)
                elif self.new_api:
                    self.logger.info("Unknown fusion pattern {} .".format(fusion_name))
                if self.remove_redundant_quant_flag:
                    self.input_graph = self.remove_redundant_quantization(self.input_graph)
                return self.input_graph, self.exclude_bn_nodes

            self.input_graph = self.output_graph
            self._reset_output_node_maps()
            if self.remove_redundant_quant_flag:
                self.output_graph = self.remove_redundant_quantization(self.output_graph)
            return self.output_graph, []
        return self.input_graph, []
