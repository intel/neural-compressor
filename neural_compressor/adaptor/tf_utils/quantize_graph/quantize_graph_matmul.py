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
import numpy as np

from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import dtypes

from .quantize_graph_common import QuantizeGraphHelper as helper
from .quantize_graph_base import QuantizeNodeBase
from tensorflow.python.framework import tensor_util

class FuseNodeStartWithMatmul(QuantizeNodeBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.sorted_patterns = sorted(self.patterns,
                                      key=lambda i: len(i),
                                      reverse=True)
        self.fusion_op_type = set(fusion[0] for fusion in self.patterns)
        self.fusion_mapping = {
            'MatMulBiasAdd': self.apply_matmul_biasadd_fusion,
            'MatMulBiasAddRelu': self.apply_matmul_biasadd_relu_fusion,
        }

    def apply_matmul_biasadd_relu_fusion(self, match_node_name):
        skip_node_name = match_node_name[1:]
        matched_node = self.node_name_mapping[match_node_name[0]]
        control_inputs, normal_inputs = self._get_node_input(matched_node.node.name)
        weight_name = normal_inputs[1]
        weight_node = self.node_name_mapping[helper.node_name_from_input(weight_name)].node

        # FIXME We only quantize the MatMul op which second input node type is const. This is a
        # workaround for RNN model like LTSM.
        if weight_node.op != 'Const':
            self.output_graph = self.input_graph
            return []

        weights_content =  tensor_util.MakeNdarray(weight_node.attr['value'].tensor)

        if np.any(np.isnan(weights_content)):
            self.output_graph = self.input_graph
            return []

        for i in self.node_name_mapping:
            if weight_node.name in self.node_name_mapping[i].output:
                self.output_graph = self.input_graph
                return []

        q_weights_name, q_weights_min_name, q_weights_max_name = \
            self._intel_cpu_quantize_weight_eightbit(
                matched_node.node.op, self.node_name_mapping[weight_name].node, self.per_channel)

        skip_node_name.append(weight_name)

        for _, node in enumerate(self.input_graph.node):
            if node.name in skip_node_name:
                pass
            elif node.name == match_node_name[0]:
                self.logger.debug("Matched node {} with input {}.".format(node.name, node.input))

                quantized_node_name = node.name + "_eightbit_quantized_mat_mul"
                bias_node_name = self.node_name_mapping[
                    match_node_name[1]].node.input[1]
                relu_node_name = match_node_name[2]
                all_input_names = self._add_eightbit_prologue_nodes(
                    matched_node.node.name)
                all_input_names = all_input_names[:1] + [q_weights_name] + all_input_names[1:]
                all_input_names.append(q_weights_min_name)
                all_input_names.append(q_weights_max_name)
                quantized_node_input_names = all_input_names[:2] + [
                    bias_node_name
                ] + all_input_names[2:] + control_inputs

                quantized_matmul_node = helper.create_node(
                    "QuantizedMatMulWithBiasAndRelu", quantized_node_name,
                    quantized_node_input_names)

                helper.copy_attr(quantized_matmul_node, "transpose_a", node.attr["transpose_a"])
                helper.copy_attr(quantized_matmul_node, "transpose_b", node.attr["transpose_b"])
                helper.set_attr_dtype(quantized_matmul_node, "T1", dtypes.quint8)
                helper.set_attr_dtype(quantized_matmul_node, "T2", dtypes.qint8)
                helper.set_attr_dtype(quantized_matmul_node, "Toutput", dtypes.qint32)
                helper.set_attr_string(quantized_matmul_node, 'input_quant_mode',
                                       b'MIN_FIRST' if self.is_asymmetric else b'SCALED')

                self.add_output_graph_node(quantized_matmul_node)

                quantize_down_name = self._add_quantize_down_nodes(
                    node, quantized_node_name, dtypes.quint8, False)
                self._intel_cpu_add_dequantize_result_node(quantize_down_name, relu_node_name)
            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)
        return match_node_name

    def apply_matmul_biasadd_fusion(self, match_node_name):
        skip_node_name = match_node_name[1:]
        matched_node = self.node_name_mapping[match_node_name[0]]
        control_inputs, normal_inputs = self._get_node_input(
            matched_node.node.name)
        weight_name = normal_inputs[1]
        weight_node = self.node_name_mapping[helper.node_name_from_input(weight_name)].node

        # FIXME We only quantize the MatMul op which second input node type is const. This is a
        # workaround for RNN model like LTSM.
        if weight_node.op != 'Const':
            self.output_graph = self.input_graph
            return []

        #TODO Remove below two lines once the TF enabled the QuantizedMatMul while
        # transpose_a/transpose_a could be set to True.
        if matched_node.node.attr["transpose_a"].b == True or \
            matched_node.node.attr["transpose_b"].b == True:
            self.output_graph = self.input_graph
            return []

        weights_content =  tensor_util.MakeNdarray(weight_node.attr['value'].tensor)

        if np.any(np.isnan(weights_content)):
            self.output_graph = self.input_graph
            return []

        for i in self.node_name_mapping:
            if weight_node.input and not weight_node.input[0].startswith('^') \
                    and weight_node.name in self.node_name_mapping[i].output:
                self.output_graph = self.input_graph
                return []

        q_weights_name, q_weights_min_name, q_weights_max_name = \
            self._intel_cpu_quantize_weight_eightbit(
                matched_node.node.op, self.node_name_mapping[weight_name].node, self.per_channel)

        skip_node_name.append(weight_name)

        for _, node in enumerate(self.input_graph.node):
            if node.name in skip_node_name:
                pass
            elif node.name == match_node_name[0]:
                self.logger.debug("Matched node {} with input {}.".format(node.name, node.input))

                quantized_node_name = node.name + "_eightbit_quantized_mat_mul"
                bias_node_name = self.node_name_mapping[match_node_name[1]].node.input[1]
                all_input_names = self._add_eightbit_prologue_nodes(matched_node.node.name)
                all_input_names = all_input_names[:1] + [q_weights_name] + all_input_names[1:]
                all_input_names.append(q_weights_min_name)
                all_input_names.append(q_weights_max_name)
                quantized_node_input_names = all_input_names[:2] + [
                    bias_node_name
                ] + all_input_names[2:] + control_inputs

                quantized_matmul_node = helper.create_node(
                    "QuantizedMatMulWithBias", quantized_node_name,
                    quantized_node_input_names)

                helper.copy_attr(quantized_matmul_node, "transpose_a", node.attr["transpose_a"])
                helper.copy_attr(quantized_matmul_node, "transpose_b", node.attr["transpose_b"])
                helper.set_attr_dtype(quantized_matmul_node, "T1", dtypes.quint8)
                helper.set_attr_dtype(quantized_matmul_node, "T2", dtypes.qint8)
                helper.set_attr_dtype(quantized_matmul_node, "Toutput", dtypes.qint32)
                helper.set_attr_dtype(quantized_matmul_node, "Tbias", dtypes.float32)
                helper.set_attr_string(quantized_matmul_node, 'input_quant_mode',
                                       b'MIN_FIRST' if self.is_asymmetric else b'SCALED')

                self.add_output_graph_node(quantized_matmul_node)
                requantize_type = dtypes.qint8

                quantize_down_name = self._add_quantize_down_nodes(
                    node, quantized_node_name, requantize_type, False)
                self._intel_cpu_add_dequantize_result_node(
                    quantize_down_name, match_node_name[1], requantize_type)
            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)
        return match_node_name

    def get_longest_fuse(self):
        self._get_op_list()
        matched_rule, _ = self._is_match(self.sorted_patterns)
        return matched_rule

    def apply_the_transform(self):
        self._get_op_list()
        matched_rule, matched_node_name = self._is_match(self.sorted_patterns)

        if matched_node_name:
            self.output_graph = graph_pb2.GraphDef()
            fusion_name = ''.join(matched_rule)
            if fusion_name in self.fusion_mapping:
                matched_nodes = self.fusion_mapping[fusion_name](matched_node_name)
            else:
                self.logger.debug("Unknown fusion pattern {}.".format(fusion_name))
                if self.remove_redundant_quant_flag:
                    self.input_graph = self.remove_redundant_quantization(self.input_graph)
                return self.input_graph, []

            self.input_graph = self.output_graph
            self._reset_output_node_maps()
            if self.remove_redundant_quant_flag:
                self.output_graph = self.remove_redundant_quantization(self.output_graph)
            return self.output_graph, matched_nodes

        return self.input_graph, []
