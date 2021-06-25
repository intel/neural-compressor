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

from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util

from .quantize_graph_common import QuantizeGraphHelper as helper
from .quantize_graph_base import QuantizeNodeBase


class FuseNodeStartWithConv2d(QuantizeNodeBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sorted_patterns = sorted(self.patterns,
                                      key=lambda i: len(i),
                                      reverse=True)
        self.fusion_mapping = {
            'Conv2DBiasAdd': self.apply_conv_biasadd_fusion,
            'Conv2DBiasAddAddNRelu': self.apply_conv_biasadd_addn_relu_fusion,
            'Conv2DBiasAddAddNRelu6': self.apply_conv_biasadd_addn_relu_fusion,
            'Conv2DBiasAddAddV2Relu': self.apply_conv_biasadd_addn_relu_fusion,
            'Conv2DBiasAddAddV2Relu6': self.apply_conv_biasadd_addn_relu_fusion,
            'Conv2DBiasAddAddRelu': self.apply_conv_biasadd_addn_relu_fusion,
            'Conv2DBiasAddRelu6': self.apply_conv_biasadd_relu_fusion,
            'Conv2DBiasAddRelu': self.apply_conv_biasadd_relu_fusion,
            'Conv2DBiasAddLeakyRelu': self.apply_conv_biasadd_relu_fusion,
            'Conv2DBiasAddLeakyReluAddV2': self.apply_conv_biasadd_addn_relu_fusion,
            'Conv2DAddRelu6': self.apply_conv_biasadd_relu_fusion,
            'Conv2DAddRelu': self.apply_conv_biasadd_relu_fusion,
            'DepthwiseConv2dNativeAddRelu6':
            self.apply_conv_biasadd_relu_fusion,
            'DepthwiseConv2dNativeBiasAddRelu6':
            self.apply_conv_biasadd_relu_fusion,
            'Conv2D': self.apply_conv_single_fusion,
            'DepthwiseConv2dNative': self.apply_conv_single_fusion,
        }

    def apply_conv_single_fusion(self, match_node_name):
        skip_node_name = match_node_name[1:]
        matched_node = self.node_name_mapping[match_node_name[0]]
        _, normal_inputs = self._get_node_input(matched_node.node.name)
        weight_name = normal_inputs[1]
        # TODO this is workaround as the tf 2.1 doesn't support depthwise/conv s8
        # feature.
        if self.enable_s8 and not self._find_relu_node(matched_node.node):
            self.output_graph = self.input_graph
            return

        q_weights_name, q_weights_min_name, q_weights_max_name = \
            self._intel_cpu_quantize_weight_eightbit(matched_node.node.op,
                                                    self.node_name_mapping[weight_name].node,
                                                    self.per_channel)

        all_input_names = self._add_eightbit_prologue_nodes(matched_node.node.name)
        all_input_names = all_input_names[:1] + [q_weights_name] + all_input_names[1:]
        all_input_names.append(q_weights_min_name)
        all_input_names.append(q_weights_max_name)
        skip_node_name.append(weight_name)

        for _, node in enumerate(self.input_graph.node):
            if node.name in skip_node_name:
                self.logger.debug("skip node {}".format(node.name))
            elif node.name == match_node_name[0]:
                postfix = "_eightbit_quantized_depthwise_conv"
                if node.op == "Conv2D":
                    postfix = "_eightbit_quantized_conv"
                quantized_node_name = node.name + postfix
                if node.op == "Conv2D":
                    quantized_conv_node = helper.create_node(
                        "QuantizedConv2DPerChannel"
                        if self.per_channel else "QuantizedConv2D",
                        quantized_node_name, all_input_names)

                elif node.op == "DepthwiseConv2dNative":
                    quantized_conv_node = helper.create_node(
                        "QuantizedDepthwiseConv2D", quantized_node_name,
                        all_input_names)

                helper.copy_attr(quantized_conv_node, "strides",
                                 node.attr["strides"])
                helper.copy_attr(quantized_conv_node, "padding",
                                 node.attr["padding"])
                if node.op != 'DepthwiseConv2dNative' and "padding_list" in node.attr:
                    helper.copy_attr(quantized_conv_node, "padding_list",
                                     node.attr["padding_list"])
                helper.copy_attr(quantized_conv_node, "dilations",
                                 node.attr["dilations"])
                input_data_type = dtypes.quint8 if self._find_relu_node(
                    node) else dtypes.qint8
                helper.set_attr_dtype(quantized_conv_node, "Tinput",
                                      input_data_type)
                helper.set_attr_dtype(quantized_conv_node, "Tfilter",
                                      dtypes.qint8)
                helper.set_attr_dtype(quantized_conv_node, "out_type",
                                      dtypes.qint32)
                self.add_output_graph_node(quantized_conv_node)
                quantize_down_name = self._add_quantize_down_nodes(
                    node, quantized_node_name, dtypes.qint8)
                self._intel_cpu_add_dequantize_result_node(
                    quantize_down_name, node.name, dtypes.qint8)
            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)

    def apply_conv_biasadd_relu_fusion(self, match_node_name):
        """Fuse the conv/biasadd/relu pattern.

        Arguments:
            match_node_name {[type]} -- [description]
        """

        skip_node_name = match_node_name[1:]
        matched_node = self.node_name_mapping[match_node_name[0]]
        control_inputs, normal_inputs = self._get_node_input(matched_node.node.name)
        weight_name = normal_inputs[1]

        q_weights_name, q_weights_min_name, q_weights_max_name = \
            self._intel_cpu_quantize_weight_eightbit(
                matched_node.node.op, self.node_name_mapping[weight_name].node, self.per_channel)

        all_input_names = self._add_eightbit_prologue_nodes(matched_node.node.name)
        all_input_names = all_input_names[:1] + [q_weights_name] + all_input_names[1:]
        all_input_names.append(q_weights_min_name)
        all_input_names.append(q_weights_max_name)
        skip_node_name.append(weight_name)

        for _, node in enumerate(self.input_graph.node):
            if node.name in skip_node_name:
                self.logger.debug("skip node {}".format(node.name))
            elif node.name == match_node_name[0]:

                self.logger.debug("apply_conv_biasadd_relu_fusion")
                postfix = "_eightbit_quantized_depthwise_conv"
                if node.op == "Conv2D":
                    postfix = "_eightbit_quantized_conv"
                quantized_node_name = node.name + postfix
                bias_node_name = self.node_name_mapping[match_node_name[1]].node.input[1]
                relu_node_name = match_node_name[2]
                is_relu6 = self.node_name_mapping[relu_node_name].node.op == "Relu6"
                quantized_node_input_names = all_input_names[:2] + \
                    [bias_node_name] + all_input_names[2:] + control_inputs
                is_leakyrelu = self.node_name_mapping[relu_node_name].node.op == "LeakyRelu"
                quantized_conv_node_op = 'QuantizedDepthwiseConv2DWithBiasAndRelu'
                if node.op == "Conv2D" or is_leakyrelu:
                    quantized_conv_node_op = "QuantizedConv2DWithBiasAndRelu"
                quantized_conv_node = helper.create_node(
                    quantized_conv_node_op,
                    quantized_node_name, quantized_node_input_names)
                helper.copy_attr(quantized_conv_node, "strides", node.attr["strides"])
                helper.copy_attr(quantized_conv_node, "padding", node.attr["padding"])
                if "alpha" in self.node_name_mapping[relu_node_name].node.attr:
                    helper.copy_attr(quantized_conv_node, "alpha",
                    self.node_name_mapping[relu_node_name].node.attr["alpha"])
                if node.op != 'DepthwiseConv2dNative' and "padding_list" in node.attr:
                    helper.copy_attr(quantized_conv_node, "padding_list",
                    node.attr["padding_list"])
                helper.copy_attr(quantized_conv_node, "dilations", node.attr["dilations"])
                input_data_type = dtypes.quint8 if self._find_relu_node(node) else dtypes.qint8
                helper.set_attr_dtype(quantized_conv_node, "Tinput", input_data_type)
                helper.set_attr_dtype(quantized_conv_node, "Tfilter",dtypes.qint8)
                helper.set_attr_dtype(quantized_conv_node, "out_type", dtypes.qint32)
                self.add_output_graph_node(quantized_conv_node)
                if not is_leakyrelu:
                    quantize_down_name = self._add_quantize_down_nodes(
                        node, quantized_node_name, dtypes.quint8, is_relu6)
                    self._intel_cpu_add_dequantize_result_node(
                        quantize_down_name, relu_node_name)
                else:
                    quantize_down_name = self._add_quantize_down_nodes(
                        node, quantized_node_name, dtypes.qint8, False)
                    self._intel_cpu_add_dequantize_result_node(
                        quantize_down_name, relu_node_name, dtype=dtypes.qint8)
            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)

    def apply_conv_biasadd_fusion(self, match_node_name):
        skip_node_name = match_node_name[1:]
        matched_node = self.node_name_mapping[match_node_name[0]]
        control_inputs, normal_inputs = self._get_node_input(
            matched_node.node.name)
        weight_name = normal_inputs[1]

        q_weights_name, q_weights_min_name, q_weights_max_name = \
            self._intel_cpu_quantize_weight_eightbit(
                matched_node.node.op, self.node_name_mapping[weight_name].node, self.per_channel)

        all_input_names = self._add_eightbit_prologue_nodes(matched_node.node.name)
        all_input_names = all_input_names[:1] + [q_weights_name] + all_input_names[1:]
        all_input_names.append(q_weights_min_name)
        all_input_names.append(q_weights_max_name)
        skip_node_name.append(weight_name)

        for _, node in enumerate(self.input_graph.node):
            if node.name in skip_node_name:
                pass
            elif node.name == match_node_name[0]:
                self.logger.debug("matched node {} with input {}".format(
                    node.name, node.input))

                self.logger.debug("apply_conv_biasadd_fusion")

                quantized_node_name = node.name + "_eightbit_quantized_conv"
                bias_node_name = self.node_name_mapping[
                    match_node_name[1]].node.input[1]
                quantized_node_input_names = all_input_names[:2] + [
                    bias_node_name
                ] + all_input_names[2:] + control_inputs

                quantized_conv_node = helper.create_node(
                    "QuantizedConv2DWithBias", quantized_node_name,
                    quantized_node_input_names)
                helper.copy_attr(quantized_conv_node, "strides",
                                 node.attr["strides"])
                helper.copy_attr(quantized_conv_node, "padding",
                                 node.attr["padding"])
                if "padding_list" in node.attr:
                    helper.copy_attr(quantized_conv_node, "padding_list",
                                     node.attr["padding_list"])
                helper.copy_attr(quantized_conv_node, "dilations",
                                 node.attr["dilations"])

                input_data_type = dtypes.quint8 if self._find_relu_node(
                    node) else dtypes.qint8
                helper.set_attr_dtype(quantized_conv_node, "Tinput",
                                      input_data_type)
                helper.set_attr_dtype(quantized_conv_node, "Tfilter",
                                      dtypes.qint8)
                helper.set_attr_dtype(quantized_conv_node, "out_type",
                                      dtypes.qint32)
                self.add_output_graph_node(quantized_conv_node)
                requantize_type = dtypes.qint8

                quantize_down_name = self._add_quantize_down_nodes(
                    node, quantized_node_name, requantize_type, False)
                self._intel_cpu_add_dequantize_result_node(
                    quantize_down_name, match_node_name[1], requantize_type)
            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)

    def apply_conv_biasadd_addn_relu_fusion(self, match_node_name):
        skip_node_name = match_node_name[1:]
        matched_node = self.node_name_mapping[match_node_name[0]]
        control_inputs, normal_inputs = self._get_node_input(
            matched_node.node.name)
        weight_name = normal_inputs[1]

        third_node = self.node_name_mapping[match_node_name[2]].node
        forth_node = self.node_name_mapping[match_node_name[3]].node
        is_leakyrelu_add_fusion = third_node.op == 'LeakyRelu' and forth_node.op.find('Add') != -1

        q_weights_name, q_weights_min_name, q_weights_max_name = \
            self._intel_cpu_quantize_weight_eightbit(
                matched_node.node.op, self.node_name_mapping[weight_name].node, self.per_channel)

        all_input_names = self._add_eightbit_prologue_nodes(matched_node.node.name)
        all_input_names = all_input_names[:1] + [q_weights_name] + all_input_names[1:]
        all_input_names.append(q_weights_min_name)
        all_input_names.append(q_weights_max_name)
        skip_node_name.append(weight_name)

        for _, node in enumerate(self.input_graph.node):
            if node.name in skip_node_name:
                self.logger.debug("skip node {}".format(node.name))
            elif node.name == match_node_name[0]:
                self.logger.debug("matched node {} with input {}".format(
                    node.name, node.input))

                self.logger.debug("apply_conv_biasadd_addn_relu_fusion")

                quantized_node_name = node.name + "_eightbit_quantized_conv"

                bias_node_name = self.node_name_mapping[match_node_name[1]].node.input[1]
                leaky_offset = 1 if is_leakyrelu_add_fusion else 0
                if is_leakyrelu_add_fusion:
                    relu_node_name = match_node_name[2]
                else:
                    relu_node_name = match_node_name[3]
                is_relu6 = self.node_name_mapping[relu_node_name].node.op == "Relu6"

                sum_index = 1 if match_node_name[1 + leaky_offset] == self.node_name_mapping[
                    match_node_name[2 + leaky_offset]].node.input[0] else 0
                quantized_node_input_names = all_input_names[:2] + [
                    bias_node_name
                ] + all_input_names[2:] + [
                    self.node_name_mapping[
                        match_node_name[2 + leaky_offset]].node.input[sum_index]
                ] + control_inputs

                node_op = "QuantizedConv2DWithBiasReluAndSum" if is_leakyrelu_add_fusion \
                            else "QuantizedConv2DWithBiasSumAndRelu"

                quantized_conv_node = helper.create_node(node_op, quantized_node_name,
                    quantized_node_input_names)
                helper.copy_attr(quantized_conv_node, "strides", node.attr["strides"])
                helper.copy_attr(quantized_conv_node, "padding", node.attr["padding"])
                if "padding_list" in node.attr:
                    helper.copy_attr(quantized_conv_node, "padding_list",
                                     node.attr["padding_list"])
                helper.copy_attr(quantized_conv_node, "dilations", node.attr["dilations"])
                input_data_type = dtypes.quint8 if self._find_relu_node(
                    node) else dtypes.qint8
                helper.set_attr_dtype(quantized_conv_node, "Tinput", input_data_type)
                helper.set_attr_dtype(quantized_conv_node, "Tfilter", dtypes.qint8)
                helper.set_attr_dtype(quantized_conv_node, "out_type", dtypes.qint32)
                if "alpha" in self.node_name_mapping[relu_node_name].node.attr:
                    helper.copy_attr(quantized_conv_node, "alpha",
                    self.node_name_mapping[relu_node_name].node.attr["alpha"])
                self.add_output_graph_node(quantized_conv_node)

                if is_leakyrelu_add_fusion:
                    quantize_down_name = self._add_quantize_down_nodes(
                                        node, quantized_node_name, dtypes.qint8, False)
                    self._intel_cpu_add_dequantize_result_node(
                        quantize_down_name, match_node_name[3], dtype=dtypes.qint8)
                else:
                    quantize_down_name = self._add_quantize_down_nodes(
                        node, quantized_node_name, dtypes.quint8, is_relu6)
                    self._intel_cpu_add_dequantize_result_node(
                        quantize_down_name, relu_node_name)

            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)

    def get_longest_fuse(self):
        self._get_op_list()

        matched_rule, matched_node_name = self._is_match(self.sorted_patterns)
        return matched_rule, matched_node_name

    def apply_the_transform(self):
        self._get_op_list()
        matched_rule, matched_node_name = self._is_match(self.sorted_patterns)
        if matched_node_name:
            self.output_graph = graph_pb2.GraphDef()
            fusion_name = ''.join(matched_rule)
            if fusion_name in self.fusion_mapping:
                if fusion_name.find('Conv2DAddRelu') != -1:
                    for input_name in self.node_name_mapping[matched_node_name[1]].node.input:
                        input_node_name = helper.node_name_from_input(input_name)
                        if input_node_name != matched_node_name[0]:
                            add_const_input_node = self.node_name_mapping[input_node_name].node
                            add_node_content = tensor_util.MakeNdarray(
                                add_const_input_node.attr["value"].tensor)
                            if add_node_content.ndim != 1:
                                fusion_name = 'Conv2D'
                                matched_node_name = matched_node_name[:1]
                self.fusion_mapping[fusion_name](matched_node_name)
            else:
                self.logger.info("Unknown match {}".format(fusion_name))
                if self.remove_redundant_quant_flag:
                    self.input_graph = self.remove_redundant_quantization(self.input_graph)
                return self.input_graph, []

            self.input_graph = self.output_graph
            self._reset_output_node_maps()
            if self.remove_redundant_quant_flag:
                self.output_graph = self.remove_redundant_quantization(self.output_graph)

            return self.output_graph, matched_node_name

        self.logger.debug("No more match, exit...")
        return self.input_graph, []
