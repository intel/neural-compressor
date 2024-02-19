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
"""Quantize Conv2DBackpropInput and Conv3DBackpropInputV2."""

import numpy as np
import tensorflow as tf
from tensorflow.core.framework import graph_pb2, node_def_pb2
from tensorflow.python.framework import dtypes, tensor_util

from neural_compressor.tensorflow.quantization.utils.quantize_graph_common import QuantizeGraphHelper as helper

from ..quantize_graph_base import QuantizeNodeBase


class FuseNodeStartWithDeconv2d(QuantizeNodeBase):
    """Quantize Conv2DBackpropInput and Conv3DBackpropInputV2 and apply the fusion."""

    exclude_deconv_nodes = []

    def __init__(self, **kwargs):
        """Initialization."""
        super().__init__(**kwargs)
        self.sorted_patterns = sorted(self.patterns, key=lambda i: len(i), reverse=True)
        if self.new_api:
            self.fusion_mapping = {
                "DequantizeConv2DBackpropInputQuantizeV2": self.apply_single_deconv2d_fusion,
                "DequantizeConv2DBackpropInputBiasAddQuantizeV2": self.apply_deconv2d_biasadd_fusion,
                "DequantizeConv3DBackpropInputV2QuantizeV2": self.apply_single_deconv3d_fusion,
                "DequantizeConv3DBackpropInputV2BiasAddQuantizeV2": self.apply_deconv3d_biasadd_fusion,
            }

    def apply_single_deconv2d_fusion(self, match_node_name):
        """Apply single Conv2DBackpropInput fusion.

        Dequantize + Conv2DBackpropInput + QuantizeV2
        """
        skip_node_name = match_node_name[2:]
        matched_node = self.node_name_mapping[match_node_name[1]]

        control_inputs, normal_inputs = self._get_node_input(matched_node.node.name)
        _, q_inputs = self._get_node_input(normal_inputs[2])
        _, q_weights_inputs = self._get_node_input(normal_inputs[1])

        quantizev2_weights_name = q_weights_inputs[0]
        _, weights_name = self._get_node_input(quantizev2_weights_name)
        weights_min_name = weights_name[1]
        weights_max_name = weights_name[2]

        q_weights_name, q_weights_min_name, q_weights_max_name = self._intel_cpu_quantize_weight_eightbit(
            matched_node.node.op, self.node_name_mapping[weights_name[0]].node, self.per_channel
        )

        all_input_names = [normal_inputs[0]] + [q_weights_name] + q_inputs[:1]
        all_input_names.append(q_weights_min_name)
        all_input_names.append(q_weights_max_name)
        all_input_names += q_inputs[1:]
        skip_node_name.append(normal_inputs[2])
        skip_node_name.append(normal_inputs[1])
        skip_node_name.append(weights_name[0])
        skip_node_name.append(weights_min_name)
        skip_node_name.append(weights_max_name)
        skip_node_name.append(quantizev2_weights_name)

        for _, node in enumerate(self.input_graph.node):
            if node.name in skip_node_name:
                self.logger.debug("skip node {}".format(node.name))
            elif node.name == match_node_name[1]:
                self.logger.debug("Matched node {} with input {}.".format(node.name, node.input))

                quantized_node_name = node.name + "_eightbit_quantized_deconv"

                quantized_node_input_names = all_input_names[:2] + all_input_names[2:] + control_inputs

                node_op = "_FusedQuantizedDeconv2D"
                quantized_deconv_node = helper.create_node(node_op, quantized_node_name, quantized_node_input_names)

                helper.copy_attr(quantized_deconv_node, "strides", node.attr["strides"])
                helper.copy_attr(quantized_deconv_node, "padding", node.attr["padding"])
                helper.copy_attr(quantized_deconv_node, "data_format", node.attr["data_format"])
                if "explicit_paddings" in node.attr:
                    helper.copy_attr(quantized_deconv_node, "explicit_paddings", node.attr["explicit_paddings"])
                helper.copy_attr(quantized_deconv_node, "dilations", node.attr["dilations"])
                input_data_type = dtypes.quint8 if self._find_relu_node(node) else dtypes.qint8
                helper.set_attr_dtype(quantized_deconv_node, "Tinput", input_data_type)
                helper.set_attr_dtype(quantized_deconv_node, "Tfilter", dtypes.qint8)
                # helper.set_attr_string(quantized_conv_node, '_kernel', b'QuantizedMklOp')
                helper.set_attr_dtype(quantized_deconv_node, "out_type", dtypes.qint32)
                # helper.set_attr_dtype(quantized_conv_node, "alpha", dtypes.quint8)

                helper.set_attr_type_list(
                    quantized_deconv_node,
                    "Thost_inputs",
                    [
                        dtypes.int32.as_datatype_enum,
                        dtypes.qint8.as_datatype_enum,
                        input_data_type.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ],
                )
                helper.set_attr_type_list(
                    quantized_deconv_node,
                    "Thost_outputs",
                    [
                        dtypes.qint32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ],
                )

                self.add_output_graph_node(quantized_deconv_node)
                quantize_down_name = self._add_quantize_down_nodes(node, quantized_node_name, dtypes.qint8, False)
                self._intel_cpu_add_dequantize_result_node(quantize_down_name, match_node_name[1], dtypes.qint8)

            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)

    def apply_deconv2d_biasadd_fusion(self, match_node_name):
        """Apply Conv2DBackpropInput BiasAdd fusion.

        Dequantize + Conv2DBackpropInput + Biasadd + QuantizeV2
        """
        skip_node_name = match_node_name[2:]
        matched_node = self.node_name_mapping[match_node_name[1]]

        second_node = self.node_name_mapping[match_node_name[2]].node
        need_insert_dummy_biasadd = 1
        add_a_node_name = helper.node_name_from_input(second_node.input[0])
        add_a_node = self.node_name_mapping[add_a_node_name].node
        add_b_node_name = helper.node_name_from_input(second_node.input[1])
        add_b_node = self.node_name_mapping[add_b_node_name].node

        # if add_a_node.op != 'Const' and add_b_node.op == 'Const':
        #      need_insert_dummy_biasadd = 0
        # if need_insert_dummy_biasadd:
        #      new_match_node_name = self._insert_dummy_biasadd(match_node_name, matched_node)
        #      # after insert dummy biasadd, that is Conv+dummybiasadd+add
        #      return self.apply_deconv2d_biasadd_fusion(new_match_node_name)

        control_inputs, normal_inputs = self._get_node_input(matched_node.node.name)
        _, q_inputs = self._get_node_input(normal_inputs[2])
        _, q_weights_inputs = self._get_node_input(normal_inputs[1])

        quantizev2_weights_name = q_weights_inputs[0]
        _, weights_name = self._get_node_input(quantizev2_weights_name)
        weights_min_name = weights_name[1]
        weights_max_name = weights_name[2]

        q_weights_name, q_weights_min_name, q_weights_max_name = self._intel_cpu_quantize_weight_eightbit(
            matched_node.node.op, self.node_name_mapping[weights_name[0]].node, self.per_channel
        )

        all_input_names = [normal_inputs[0]] + [q_weights_name] + q_inputs[:1]
        all_input_names.append(q_weights_min_name)
        all_input_names.append(q_weights_max_name)
        all_input_names += q_inputs[1:]
        skip_node_name.append(normal_inputs[2])
        skip_node_name.append(normal_inputs[1])
        skip_node_name.append(weights_name[0])
        skip_node_name.append(weights_min_name)
        skip_node_name.append(weights_max_name)
        skip_node_name.append(quantizev2_weights_name)

        for _, node in enumerate(self.input_graph.node):
            if node.name in skip_node_name:
                self.logger.debug("skip node {}".format(node.name))
            elif node.name == match_node_name[1]:
                self.logger.debug("Matched node {} with input {}.".format(node.name, node.input))

                quantized_node_name = node.name + "_eightbit_quantized_deconv"

                bias_node_name = self.node_name_mapping[match_node_name[2]].node.input[1]
                quantized_node_input_names = (
                    all_input_names[:3] + [bias_node_name] + all_input_names[3:] + control_inputs
                )

                node_op = "_FusedQuantizedDeconv2D"
                quantized_deconv_node = helper.create_node(node_op, quantized_node_name, quantized_node_input_names)

                helper.copy_attr(quantized_deconv_node, "strides", node.attr["strides"])
                helper.copy_attr(quantized_deconv_node, "padding", node.attr["padding"])
                helper.copy_attr(quantized_deconv_node, "data_format", node.attr["data_format"])
                if "explicit_paddings" in node.attr:
                    helper.copy_attr(quantized_deconv_node, "explicit_paddings", node.attr["explicit_paddings"])
                helper.copy_attr(quantized_deconv_node, "dilations", node.attr["dilations"])
                input_data_type = dtypes.quint8 if self._find_relu_node(node) else dtypes.qint8
                helper.set_attr_dtype(quantized_deconv_node, "Tinput", input_data_type)
                helper.set_attr_dtype(quantized_deconv_node, "Tfilter", dtypes.qint8)
                # helper.set_attr_string(quantized_conv_node, '_kernel', b'QuantizedMklOp')
                helper.set_attr_dtype(quantized_deconv_node, "out_type", dtypes.qint32)
                # helper.set_attr_dtype(quantized_conv_node, "alpha", dtypes.quint8)
                helper.set_attr_dtype(quantized_deconv_node, "Tbias", dtypes.float32)
                # if self.device == 'gpu' else dtypes.qint32)
                helper.set_attr_string_list(quantized_deconv_node, "fused_ops", [b"BiasAdd"])

                helper.set_attr_type_list(
                    quantized_deconv_node,
                    "Thost_inputs",
                    [
                        dtypes.int32.as_datatype_enum,
                        dtypes.qint8.as_datatype_enum,
                        input_data_type.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,  # if self.device == 'gpu' else dtypes.qint32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ],
                )
                helper.set_attr_type_list(
                    quantized_deconv_node,
                    "Thost_outputs",
                    [
                        dtypes.qint32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ],
                )

                self.add_output_graph_node(quantized_deconv_node)
                quantize_down_name = self._add_quantize_down_nodes(node, quantized_node_name, dtypes.qint8, False)
                self._intel_cpu_add_dequantize_result_node(quantize_down_name, match_node_name[2], dtypes.qint8)

            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)

    def apply_single_deconv3d_fusion(self, match_node_name):
        """Apply single Conv3DBackpropInputV2 fusion.

        Dequantize + Conv3DBackpropInputV2 + QuantizeV2
        """
        skip_node_name = match_node_name[2:]
        matched_node = self.node_name_mapping[match_node_name[1]]

        control_inputs, normal_inputs = self._get_node_input(matched_node.node.name)
        _, q_inputs = self._get_node_input(normal_inputs[2])
        _, q_weights_inputs = self._get_node_input(normal_inputs[1])

        quantizev2_weights_name = q_weights_inputs[0]
        _, weights_name = self._get_node_input(quantizev2_weights_name)
        weights_min_name = weights_name[1]
        weights_max_name = weights_name[2]

        q_weights_name, q_weights_min_name, q_weights_max_name = self._intel_cpu_quantize_weight_eightbit(
            matched_node.node.op, self.node_name_mapping[weights_name[0]].node, self.per_channel
        )

        all_input_names = [normal_inputs[0]] + [q_weights_name] + q_inputs[:1]
        all_input_names.append(q_weights_min_name)
        all_input_names.append(q_weights_max_name)
        all_input_names += q_inputs[1:]
        skip_node_name.append(normal_inputs[2])
        skip_node_name.append(normal_inputs[1])
        skip_node_name.append(weights_name[0])
        skip_node_name.append(weights_min_name)
        skip_node_name.append(weights_max_name)
        skip_node_name.append(quantizev2_weights_name)

        for _, node in enumerate(self.input_graph.node):
            if node.name in skip_node_name:
                self.logger.debug("skip node {}".format(node.name))
            elif node.name == match_node_name[1]:
                self.logger.debug("Matched node {} with input {}.".format(node.name, node.input))

                quantized_node_name = node.name + "_eightbit_quantized_deconv"

                quantized_node_input_names = all_input_names[:2] + all_input_names[2:] + control_inputs

                node_op = "_FusedQuantizedDeconv3D"
                quantized_deconv_node = helper.create_node(node_op, quantized_node_name, quantized_node_input_names)

                helper.copy_attr(quantized_deconv_node, "strides", node.attr["strides"])
                helper.copy_attr(quantized_deconv_node, "padding", node.attr["padding"])
                helper.copy_attr(quantized_deconv_node, "data_format", node.attr["data_format"])
                if "explicit_paddings" in node.attr:
                    helper.copy_attr(quantized_deconv_node, "explicit_paddings", node.attr["explicit_paddings"])
                helper.copy_attr(quantized_deconv_node, "dilations", node.attr["dilations"])
                input_data_type = dtypes.quint8 if self._find_relu_node(node) else dtypes.qint8
                helper.set_attr_dtype(quantized_deconv_node, "Tinput", input_data_type)
                helper.set_attr_dtype(quantized_deconv_node, "Tfilter", dtypes.qint8)
                # helper.set_attr_string(quantized_conv_node, '_kernel', b'QuantizedMklOp')
                helper.set_attr_dtype(quantized_deconv_node, "out_type", dtypes.qint32)
                # helper.set_attr_dtype(quantized_conv_node, "alpha", dtypes.quint8)

                helper.set_attr_type_list(
                    quantized_deconv_node,
                    "Thost_inputs",
                    [
                        dtypes.int32.as_datatype_enum,
                        dtypes.qint8.as_datatype_enum,
                        input_data_type.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ],
                )
                helper.set_attr_type_list(
                    quantized_deconv_node,
                    "Thost_outputs",
                    [
                        dtypes.qint32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ],
                )

                self.add_output_graph_node(quantized_deconv_node)
                quantize_down_name = self._add_quantize_down_nodes(node, quantized_node_name, dtypes.qint8, False)
                self._intel_cpu_add_dequantize_result_node(quantize_down_name, match_node_name[1], dtypes.qint8)

            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)

    def apply_deconv3d_biasadd_fusion(self, match_node_name):
        """Apply the Conv3DBackpropInputV2 BiasAdd fusion.

        Dequantize + Conv3DBackpropInputV2 + Biasadd + QuantizeV2
        """
        skip_node_name = match_node_name[2:]
        matched_node = self.node_name_mapping[match_node_name[1]]

        control_inputs, normal_inputs = self._get_node_input(matched_node.node.name)
        _, q_inputs = self._get_node_input(normal_inputs[2])
        _, q_weights_inputs = self._get_node_input(normal_inputs[1])

        quantizev2_weights_name = q_weights_inputs[0]
        _, weights_name = self._get_node_input(quantizev2_weights_name)
        weights_min_name = weights_name[1]
        weights_max_name = weights_name[2]

        q_weights_name, q_weights_min_name, q_weights_max_name = self._intel_cpu_quantize_weight_eightbit(
            matched_node.node.op, self.node_name_mapping[weights_name[0]].node, self.per_channel
        )

        all_input_names = [normal_inputs[0]] + [q_weights_name] + q_inputs[:1]
        all_input_names.append(q_weights_min_name)
        all_input_names.append(q_weights_max_name)
        all_input_names += q_inputs[1:]
        skip_node_name.append(normal_inputs[2])
        skip_node_name.append(normal_inputs[1])
        skip_node_name.append(weights_name[0])
        skip_node_name.append(weights_min_name)
        skip_node_name.append(weights_max_name)
        skip_node_name.append(quantizev2_weights_name)

        for _, node in enumerate(self.input_graph.node):
            if node.name in skip_node_name:
                self.logger.debug("skip node {}".format(node.name))
            elif node.name == match_node_name[1]:
                self.logger.debug("Matched node {} with input {}.".format(node.name, node.input))

                quantized_node_name = node.name + "_eightbit_quantized_deconv"

                bias_node_name = self.node_name_mapping[match_node_name[2]].node.input[1]
                quantized_node_input_names = (
                    all_input_names[:3] + [bias_node_name] + all_input_names[3:] + control_inputs
                )

                node_op = "_FusedQuantizedDeconv3D"
                quantized_deconv_node = helper.create_node(node_op, quantized_node_name, quantized_node_input_names)

                helper.copy_attr(quantized_deconv_node, "strides", node.attr["strides"])
                helper.copy_attr(quantized_deconv_node, "padding", node.attr["padding"])
                helper.copy_attr(quantized_deconv_node, "data_format", node.attr["data_format"])
                if "explicit_paddings" in node.attr:
                    helper.copy_attr(quantized_deconv_node, "explicit_paddings", node.attr["explicit_paddings"])
                helper.copy_attr(quantized_deconv_node, "dilations", node.attr["dilations"])
                input_data_type = dtypes.quint8 if self._find_relu_node(node) else dtypes.qint8
                helper.set_attr_dtype(quantized_deconv_node, "Tinput", input_data_type)
                helper.set_attr_dtype(quantized_deconv_node, "Tfilter", dtypes.qint8)
                # helper.set_attr_string(quantized_conv_node, '_kernel', b'QuantizedMklOp')
                helper.set_attr_dtype(quantized_deconv_node, "out_type", dtypes.qint32)
                # helper.set_attr_dtype(quantized_conv_node, "alpha", dtypes.quint8)
                helper.set_attr_dtype(quantized_deconv_node, "Tbias", dtypes.float32)
                # if self.device == 'gpu' else dtypes.qint32)
                helper.set_attr_string_list(quantized_deconv_node, "fused_ops", [b"BiasAdd"])

                helper.set_attr_type_list(
                    quantized_deconv_node,
                    "Thost_inputs",
                    [
                        dtypes.int32.as_datatype_enum,
                        dtypes.qint8.as_datatype_enum,
                        input_data_type.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,  # if self.device == 'gpu' else dtypes.qint32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ],
                )
                helper.set_attr_type_list(
                    quantized_deconv_node,
                    "Thost_outputs",
                    [
                        dtypes.qint32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ],
                )

                self.add_output_graph_node(quantized_deconv_node)
                quantize_down_name = self._add_quantize_down_nodes(node, quantized_node_name, dtypes.qint8, False)
                self._intel_cpu_add_dequantize_result_node(quantize_down_name, match_node_name[2], dtypes.qint8)

            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)

    def get_longest_fuse(self):
        """Get the longest fusion pattern."""
        self._get_op_list()

        matched_rule, matched_node_name = self._is_match_deconv(self.sorted_patterns)
        return matched_rule, matched_node_name

    def apply_the_transform(self):
        """Quantize Conv2DBackpropInput and Conv3DBackpropInputV2 and apply the fusion."""
        self._get_op_list()
        matched_rule, matched_node_name = self._is_match_deconv(self.sorted_patterns, True)
        if matched_node_name:
            self.output_graph = graph_pb2.GraphDef()
            fusion_name = "".join(matched_rule)
            if fusion_name in self.fusion_mapping:
                self.fusion_mapping[fusion_name](matched_node_name)
            else:  # pragma: no cover
                self.logger.info("Unknown fusion pattern {}.".format(fusion_name))
                if self.remove_redundant_quant_flag:
                    self.input_graph = self.remove_redundant_quantization(self.input_graph)
                return self.input_graph, self.exclude_deconv_nodes

            self.input_graph = self.output_graph
            self._reset_output_node_maps()
            if self.remove_redundant_quant_flag:
                self.output_graph = self.remove_redundant_quantization(self.output_graph)

            return self.output_graph, self.exclude_deconv_nodes

        if self.remove_redundant_quant_flag:
            self.input_graph = self.remove_redundant_quantization(self.input_graph)
        return self.input_graph, self.exclude_deconv_nodes

    def _is_match_deconv(self, patterns, qdq_inserted=False):
        """Detect the rule matched nodes collections.

        Returns:
            [List] -- [the matched rule]
            [String] -- [the list contains the matched node name]
        """
        matched_node_name = []

        for k, v in enumerate(self.op_list):
            if v in set(fusion[1] for fusion in patterns):
                cur_node = self.node_name_mapping[list(self.node_name_mapping.keys())[k]].node
                if cur_node.name != self.start_node_name:
                    continue

                if qdq_inserted:
                    _, normal_inputs = self._get_node_input(cur_node.name)

                for sub_rule in patterns:
                    if sub_rule[0] != "Dequantize" or sub_rule[-1] != "QuantizeV2":
                        self.exclude_deconv_nodes.append(cur_node.name)
                        continue
                    if v != sub_rule[1]:
                        self.exclude_deconv_nodes.append(cur_node.name)
                        continue

                    if qdq_inserted:
                        input_index = 0
                        if sub_rule[1] in ("Conv2DBackpropInput", "Conv3DBackpropInputV2"):
                            input_index = 2
                        if (
                            self.node_name_mapping[normal_inputs[input_index]].node.op != "Dequantize"
                            or self.node_name_mapping[normal_inputs[1]].node.op != "Dequantize"
                        ):
                            continue

                    sub_rule_len = len(sub_rule) - 2
                    self.logger.debug("Try to apply rule: {}".format(sub_rule))

                    cur_node_name = list(self.node_name_mapping.keys())[k]

                    matched_node_name.clear()
                    matched_node_name.append(sub_rule[0])
                    matched_node_name.append(cur_node_name)

                    while sub_rule_len > 1:
                        if not self.node_name_mapping[cur_node_name].output:
                            self.logger.debug("Fail to match {}".format(sub_rule))
                            break

                        next_node_name = self.node_name_mapping[cur_node_name].output[0]

                        is_shared_output = True if len(self.node_name_mapping[cur_node_name].output) > 1 else False

                        next_node_op = self.node_name_mapping[next_node_name].node.op
                        if next_node_op == sub_rule[-sub_rule_len]:
                            if not is_shared_output:
                                matched_node_name.append(next_node_name)
                                sub_rule_len -= 1
                                cur_node_name = next_node_name
                            else:
                                matched_node_name.clear()
                                self.logger.debug("Fail to match {}.".format(sub_rule))
                                break
                        else:
                            matched_node_name.clear()
                            self.logger.debug("Fail to match {}.".format(sub_rule))
                            break

                    if sub_rule_len == 1:
                        matched_node_name.append(sub_rule[-1])
                        self.logger.debug("Match {} on nodes {}.".format(sub_rule, matched_node_name))
                        return sub_rule, matched_node_name

        return None, None
