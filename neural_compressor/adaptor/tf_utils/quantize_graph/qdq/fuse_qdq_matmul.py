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
import numpy as np

from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import dtypes

from neural_compressor.adaptor.tf_utils.quantize_graph_common import QuantizeGraphHelper as helper
from ..quantize_graph_base import QuantizeNodeBase
from tensorflow.python.framework import tensor_util

class FuseNodeStartWithMatmul(QuantizeNodeBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.sorted_patterns = sorted(self.patterns,
                                      key=lambda i: len(i),
                                      reverse=True)
        self.fusion_op_type = set(fusion[1] for fusion in self.patterns)
        self.fusion_mapping = {
            'DequantizeMatMulBiasAddQuantizeV2': self.apply_matmul_biasadd_fusion,
            'DequantizeMatMulQuantizeV2': self.apply_matmul_biasadd_fusion,
            'DequantizeMatMulBiasAddReluQuantizeV2': self.apply_matmul_biasadd_relu_fusion,
            'DequantizeMatMulReluQuantizeV2': self.apply_matmul_biasadd_relu_fusion,
            'DequantizeBatchMatMulV2': self.apply_batchmatmulv2_fusion,
            'DequantizeBatchMatMulV2MulAdd': self.apply_batchmatmulv2_mul_add_fusion
        }


    def apply_matmul_biasadd_relu_fusion(self, match_node_name):
        # Dequantize + MatMul + BiasAdd + Relu + QuantizeV2
        matched_node = self.node_name_mapping[match_node_name[1]]
        control_inputs, normal_inputs = self._get_node_input(matched_node.node.name)

        _, q_inputs = self._get_node_input(normal_inputs[0])
        _, q_weights_inputs = self._get_node_input(normal_inputs[1])
        quantizev2_weights_name = q_weights_inputs[0]

        _, weights_name = self._get_node_input(quantizev2_weights_name)
        weights_min_name = weights_name[1]
        weights_max_name = weights_name[2]
        weight_node = self.node_name_mapping[helper.node_name_from_input(weights_name[0])].node

        # FIXME We only quantize the MatMul op which second input node type is const. This is a
        # workaround for RNN model like LTSM.
        if weight_node.op != 'Const':
            self.output_graph = self.input_graph
            return []

        weights_content = tensor_util.MakeNdarray(weight_node.attr['value'].tensor)

        if np.any(np.isnan(weights_content)):
            self.output_graph = self.input_graph
            return []

        for i in self.node_name_mapping:
            if weight_node.name in self.node_name_mapping[i].output:
                self.output_graph = self.input_graph
                return []
        second_node = self.node_name_mapping[match_node_name[2]].node
        skip_node_name = match_node_name[2:]

        need_insert_dummy_biasadd = 1
        offset = 1
        if len(match_node_name) == 5:
             add_a_node_name = helper.node_name_from_input(second_node.input[0])
             add_a_node = self.node_name_mapping[add_a_node_name].node
             add_b_node_name = helper.node_name_from_input(second_node.input[1])
             add_b_node = self.node_name_mapping[add_b_node_name].node
             if add_a_node.op != 'Const' and add_b_node.op == 'Const':
                 need_insert_dummy_biasadd = 0
                 offset = 0
             if need_insert_dummy_biasadd:
                 self.apply_matmul_biasadd_fusion(match_node_name[:2]+[match_node_name[-1]])
                 return match_node_name[1:2]

        q_weights_name, q_weights_min_name, q_weights_max_name = \
            self._intel_cpu_quantize_weight_eightbit(
                matched_node.node.op, self.node_name_mapping[weights_name[0]].node, self.per_channel)

        skip_node_name.append(normal_inputs[0])
        skip_node_name.append(normal_inputs[1])
        skip_node_name.append(weights_name[0])
        skip_node_name.append(weights_min_name)
        skip_node_name.append(weights_max_name)
        skip_node_name.append(quantizev2_weights_name)

        for _, node in enumerate(self.input_graph.node):
            if node.name in skip_node_name:
                pass
            elif node.name == match_node_name[1]:
                self.logger.debug("Matched node {} with input {}.".format(node.name, node.input))

                quantized_node_name = node.name + "_eightbit_quantized_mat_mul"
                if need_insert_dummy_biasadd:
                    t_b_index = 0 if matched_node.node.attr['transpose_b'].b else 1
                    bias_size = weights_content.shape[t_b_index]
                    bias_node_name = node.name + "_fake_bias"
                    bias_node = helper.create_constant_node(
                        bias_node_name, [0] * bias_size, dtypes.float32, shape=[bias_size]
                    )
                    self.add_output_graph_node(bias_node)
                else:
                    bias_node_name = self.node_name_mapping[match_node_name[2]].node.input[1]
                relu_node_name = match_node_name[3-offset]
                all_input_names = q_inputs[:1] + [q_weights_name] + q_inputs[1:]
                all_input_names.append(q_weights_min_name)
                all_input_names.append(q_weights_max_name)
                quantized_node_input_names = all_input_names[:2] + [
                    bias_node_name
                ] + all_input_names[2:] + control_inputs

                quantized_matmul_node = helper.create_node(
                    "QuantizedMatMulWithBiasAndRelu", quantized_node_name,
                    quantized_node_input_names)
                # quantized_matmul_node = helper.create_node(
                #     "_QuantizedFusedMatMul", quantized_node_name,
                #     quantized_node_input_names)

                helper.copy_attr(quantized_matmul_node, "transpose_a", node.attr["transpose_a"])
                helper.copy_attr(quantized_matmul_node, "transpose_b", node.attr["transpose_b"])
                helper.set_attr_dtype(quantized_matmul_node, "T1", dtypes.quint8)
                helper.set_attr_dtype(quantized_matmul_node, "T2", dtypes.qint8)
                helper.set_attr_dtype(quantized_matmul_node, "Toutput", dtypes.qint32)
                helper.set_attr_string(quantized_matmul_node, 'input_quant_mode',
                                       b'MIN_FIRST' if self.is_asymmetric else b'SCALED')
                # helper.set_attr_string_list(quantized_matmul_node, 'fused_ops', [b'BiasAdd', b'Relu'])
                # helper.set_attr_int(quantized_matmul_node, 'num_args', 1)
                # helper.set_attr_dtype(quantized_matmul_node, 'Targs', dtypes.float32)

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
        # Dequantize + MatMul + BiasAdd + QuantizeV2
        skip_node_name = match_node_name[2:]
        matched_node = self.node_name_mapping[match_node_name[1]]
        control_inputs, normal_inputs = self._get_node_input(
            matched_node.node.name)

        _, q_inputs = self._get_node_input(normal_inputs[0])
        _, q_weights_inputs = self._get_node_input(normal_inputs[1])
        quantizev2_weights_name = q_weights_inputs[0]

        _, weights_name = self._get_node_input(quantizev2_weights_name)
        weights_min_name = weights_name[1]
        weights_max_name = weights_name[2]

        weight_node = self.node_name_mapping[helper.node_name_from_input(weights_name[0])].node

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

        weights_content = tensor_util.MakeNdarray(weight_node.attr['value'].tensor)

        if np.any(np.isnan(weights_content)):
            self.output_graph = self.input_graph
            return []

        for i in self.node_name_mapping:
            if weight_node.input and not weight_node.input[0].startswith('^') \
                    and weight_node.name in self.node_name_mapping[i].output:
                self.output_graph = self.input_graph
                return []

        is_shared_output = True if len(
              matched_node.output
        ) > 1 else False

        need_insert_dummy_biasadd = 1
        if len(match_node_name) == 3:
            if is_shared_output:
                self.output_graph = self.input_graph
                return []
        else:
            second_node = self.node_name_mapping[match_node_name[2]].node
            add_a_node_name = helper.node_name_from_input(second_node.input[0])
            add_a_node = self.node_name_mapping[add_a_node_name].node
            add_b_node_name = helper.node_name_from_input(second_node.input[1])
            add_b_node = self.node_name_mapping[add_b_node_name].node
            if add_a_node.op != 'Const' and add_b_node.op == 'Const':
                 need_insert_dummy_biasadd = 0
            if need_insert_dummy_biasadd:
                 self.apply_matmul_biasadd_fusion(match_node_name[:2]+[match_node_name[-1]])
                 return match_node_name[1:2]

        q_weights_name, q_weights_min_name, q_weights_max_name = \
            self._intel_cpu_quantize_weight_eightbit(
                matched_node.node.op, self.node_name_mapping[weights_name[0]].node, self.per_channel)

        skip_node_name.append(normal_inputs[0])
        skip_node_name.append(normal_inputs[1])
        skip_node_name.append(weights_name[0])
        skip_node_name.append(weights_min_name)
        skip_node_name.append(weights_max_name)
        skip_node_name.append(quantizev2_weights_name)

        for _, node in enumerate(self.input_graph.node):
            if node.name in skip_node_name:
                pass
            elif node.name == match_node_name[1]:
                self.logger.debug("Matched node {} with input {}.".format(node.name, node.input))

                quantized_node_name = node.name + "_eightbit_quantized_mat_mul"
                if need_insert_dummy_biasadd:
                    t_b_index = 0 if matched_node.node.attr['transpose_b'].b else 1
                    bias_size = weights_content.shape[t_b_index]
                    bias_node_name = node.name + "_fake_bias"
                    bias_node = helper.create_constant_node(
                        bias_node_name, [0] * bias_size, dtypes.float32, shape=[bias_size]
                    )
                    self.add_output_graph_node(bias_node)
                else:
                    bias_node_name = self.node_name_mapping[match_node_name[2]].node.input[1]

                all_input_names = q_inputs[:1] + [q_weights_name] + q_inputs[1:]
                all_input_names.append(q_weights_min_name)
                all_input_names.append(q_weights_max_name)

                quantized_node_input_names = all_input_names[:2] + [
                    bias_node_name
                ] + all_input_names[2:] + control_inputs

                quantized_matmul_node = helper.create_node(
                    "QuantizedMatMulWithBias", quantized_node_name,
                    quantized_node_input_names)
                # quantized_matmul_node = helper.create_node(
                #     "_QuantizedFusedMatMul", quantized_node_name,
                #     quantized_node_input_names)

                helper.copy_attr(quantized_matmul_node, "transpose_a", node.attr["transpose_a"])
                helper.copy_attr(quantized_matmul_node, "transpose_b", node.attr["transpose_b"])
                helper.set_attr_dtype(quantized_matmul_node, "T1", dtypes.quint8)
                helper.set_attr_dtype(quantized_matmul_node, "T2", dtypes.qint8)
                helper.set_attr_dtype(quantized_matmul_node, "Toutput", dtypes.qint32)
                helper.set_attr_dtype(quantized_matmul_node, "Tbias", dtypes.float32)
                helper.set_attr_string(quantized_matmul_node, 'input_quant_mode',
                                       b'MIN_FIRST' if self.is_asymmetric else b'SCALED')
                # helper.set_attr_string_list(quantized_matmul_node, 'fused_ops', [b'BiasAdd'])
                # helper.set_attr_int(quantized_matmul_node, 'num_args', 1)
                # helper.set_attr_dtype(quantized_matmul_node, 'Targs', dtypes.float32)

                self.add_output_graph_node(quantized_matmul_node)
                requantize_type = dtypes.qint8

                quantize_down_name = self._add_quantize_down_nodes(
                    node, quantized_node_name, requantize_type, False)
                self._intel_cpu_add_dequantize_result_node(
                    quantize_down_name, match_node_name[1] if need_insert_dummy_biasadd else \
					match_node_name[2], requantize_type)
            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)
        return match_node_name

    def apply_batchmatmulv2_fusion(self, match_node_name): # pragma: no cover
        # Dequantize + BatchMatMulV2
        skip_node_name = match_node_name[2:]
        matched_node = self.node_name_mapping[match_node_name[1]]
        control_inputs, normal_inputs = self._get_node_input(
            matched_node.node.name)

        _, q_inputs = self._get_node_input(normal_inputs[0])
        _, q_weights_inputs = self._get_node_input(normal_inputs[1])

        quantizev2_input_name = q_inputs[0]

        quantizev2_weights_name = q_weights_inputs[0]
        _, weights_name = self._get_node_input(quantizev2_weights_name)

        weight_node = self.node_name_mapping[helper.node_name_from_input(weights_name[0])].node

        # FIXME We only quantize the MatMul op which second input node type is const. This is a
        # workaround for RNN model like LTSM.
        if weight_node.op != 'Const':
            self.output_graph = self.input_graph
            return []

        weights_content = tensor_util.MakeNdarray(weight_node.attr['value'].tensor)

        if np.any(np.isnan(weights_content)):
            self.output_graph = self.input_graph
            return []

        for i in self.node_name_mapping:
            if weight_node.input and not weight_node.input[0].startswith('^') \
                    and weight_node.name in self.node_name_mapping[i].output:
                self.output_graph = self.input_graph
                return []

        skip_node_name.append(normal_inputs[0])
        skip_node_name.append(normal_inputs[1])

        for _, node in enumerate(self.input_graph.node):
            node_control_inputs, node_normal_inputs = self._get_node_input(node.name)
            if node.name in skip_node_name:
                pass
            elif node.name == match_node_name[1]:
                self.logger.debug("Matched node {} with input {}.".format(node.name, node.input))

                quantized_node_name = node.name + "_eightbit_quantized_batch_matmul_v2"
                all_input_names = [quantizev2_input_name] + [quantizev2_weights_name] + \
                                  q_inputs[1:] + q_weights_inputs[1:]

                quantized_node_input_names = all_input_names + control_inputs

                quantized_matmul_node = helper.create_node(
                    "_QuantizedBatchMatMulV2AndDequantize", quantized_node_name,
                    quantized_node_input_names)

                helper.copy_attr(quantized_matmul_node, "adj_x", node.attr["adj_x"])
                helper.copy_attr(quantized_matmul_node, "adj_y", node.attr["adj_y"])
                helper.set_attr_dtype(quantized_matmul_node, "T1", dtypes.qint8)
                helper.set_attr_dtype(quantized_matmul_node, "T2", dtypes.qint8)
                helper.set_attr_dtype(quantized_matmul_node, "Toutput", dtypes.float32)

                self.add_output_graph_node(quantized_matmul_node)
            else:
                new_node = node_def_pb2.NodeDef()
                if node_normal_inputs and \
                   self.node_name_mapping[node_normal_inputs[0]].node.op == "BatchMatMulV2":
                    node.input[0] = "BatchMatMulV2_eightbit_quantized_batch_matmul_v2"
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)
 
        return match_node_name

    def apply_batchmatmulv2_mul_add_fusion(self, match_node_name): # pragma: no cover
        # Dequantize + BatchMatMulV2 + Mul + Add
        skip_node_name = match_node_name[2:]
        matched_node = self.node_name_mapping[match_node_name[1]]
        control_inputs, normal_inputs = self._get_node_input(
            matched_node.node.name)

        _, q_inputs = self._get_node_input(normal_inputs[0])
        _, q_weights_inputs = self._get_node_input(normal_inputs[1])
        quantizev2_weights_name = q_weights_inputs[0]
        _, weights_name = self._get_node_input(quantizev2_weights_name)

        weight_node = self.node_name_mapping[helper.node_name_from_input(weights_name[0])].node

        # FIXME We only quantize the MatMul op which second input node type is const. This is a
        # workaround for RNN model like LTSM.
        if weight_node.op != 'Const':
            self.output_graph = self.input_graph
            return []

        weights_content = tensor_util.MakeNdarray(weight_node.attr['value'].tensor)

        if np.any(np.isnan(weights_content)):
            self.output_graph = self.input_graph
            return []

        for i in self.node_name_mapping:
            if weight_node.input and not weight_node.input[0].startswith('^') \
                    and weight_node.name in self.node_name_mapping[i].output:
                self.output_graph = self.input_graph
                return []

        skip_node_name.append(normal_inputs[0])
        skip_node_name.append(normal_inputs[1])

        for _, node in enumerate(self.input_graph.node):
            node_control_inputs, node_normal_inputs = self._get_node_input(node.name)
            if node.name in skip_node_name:
                pass
            elif node.name == match_node_name[1]:
                self.logger.debug("Matched node {} with input {}.".format(node.name, node.input))

                quantized_node_name = node.name + "_eightbit_quantized_batch_matmul_v2"
                mul_node_name = self.node_name_mapping[match_node_name[2]].node.input[1]
                add_node_name = self.node_name_mapping[match_node_name[3]].node.input[1]
                skip_node_name.append(match_node_name[2])
                skip_node_name.append(match_node_name[3])
                all_input_names = q_inputs[:1] + [quantizev2_weights_name] + [mul_node_name] \
                                  + [add_node_name] + q_inputs[1:] + q_weights_inputs[1:]
                quantized_node_input_names = all_input_names + control_inputs

                quantized_matmul_node = helper.create_node(
                    "_QuantizedFusedBatchMatMulV2AndDequantize", quantized_node_name,
                    quantized_node_input_names)

                helper.copy_attr(quantized_matmul_node, "adj_x", node.attr["adj_x"])
                helper.copy_attr(quantized_matmul_node, "adj_y", node.attr["adj_y"])
                helper.set_attr_dtype(quantized_matmul_node, "T1", dtypes.qint8)
                helper.set_attr_dtype(quantized_matmul_node, "T2", dtypes.qint8)
                helper.set_attr_dtype(quantized_matmul_node, "T", dtypes.float32)
                helper.set_attr_dtype(quantized_matmul_node, "Toutput", dtypes.float32)
                helper.set_attr_string_list(quantized_matmul_node, 'fused_ops', [b'Mul', b'Add'])
                helper.set_attr_int(quantized_matmul_node, 'num_args', 2)

                self.add_output_graph_node(quantized_matmul_node)
            else:
                new_node = node_def_pb2.NodeDef()
                if node_normal_inputs and \
                   self.node_name_mapping[node_normal_inputs[0]].node.op == "Add":
                   node.input[0] = "BatchMatMulV2_eightbit_quantized_batch_matmul_v2"
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)
        return match_node_name

    def get_longest_fuse(self):
        self._get_op_list()
        matched_rule, matched_node_name = self._is_match_matmul(self.sorted_patterns)
        return matched_rule, matched_node_name

    def apply_the_transform(self):
        self._get_op_list()
        matched_rule, matched_node_name = self._is_match_matmul(self.sorted_patterns, True)
        if matched_node_name:
            _, normal_inputs = self._get_node_input(matched_node_name[1])
        if matched_node_name and self.node_name_mapping[normal_inputs[0]].node.op == matched_node_name[0]:
            self.output_graph = graph_pb2.GraphDef()
            fusion_name = ''.join(matched_rule)
            if fusion_name in self.fusion_mapping:
                matched_nodes = self.fusion_mapping[fusion_name](matched_node_name)
            else: # pragma: no cover
                self.logger.debug("Unknown fusion pattern {}.".format(fusion_name))
                if self.remove_redundant_quant_flag:
                    self.input_graph = self.remove_redundant_quantization(self.input_graph)
                return self.input_graph

            self.input_graph = self.output_graph
            self._reset_output_node_maps()
            if self.remove_redundant_quant_flag:
                self.output_graph = self.remove_redundant_quantization(self.output_graph)
            return self.output_graph

        if self.remove_redundant_quant_flag:
            self.input_graph = self.remove_redundant_quantization(self.input_graph)
        return self.input_graph

    def _is_match_matmul(self, patterns, qdq_inserted=False):
        """Detect the rule matched nodes collections.

        Returns:
            [List] -- [the matched rule]
            [String] -- [the list contains the matched node name]
        """
        matched_node_name = []

        for k, v in enumerate(self.op_list):
            if v in set(fusion[1] for fusion in patterns):
                cur_node = self.node_name_mapping[list(
                    self.node_name_mapping.keys())[k]].node
                if cur_node.name != self.start_node_name:
                    continue

                _, normal_inputs = self._get_node_input(cur_node.name)
                if not qdq_inserted:
                    weight_name = normal_inputs[1]
                    weight_node = self.node_name_mapping[helper.node_name_from_input(weight_name)].node
                    # FIXME We only quantize the MatMul op which second input node type is const. This is a
                    # workaround for RNN model like LTSM.
                    if weight_node.op != 'Const':
                        continue

                    #TODO Remove below two lines once the TF enabled the QuantizedMatMul while
                    # transpose_a/transpose_a could be set to True.
                    if cur_node.attr["transpose_a"].b == True or \
                        cur_node.attr["transpose_b"].b == True:
                        continue

                    weights_content =  tensor_util.MakeNdarray(weight_node.attr['value'].tensor)
                    if np.any(np.isnan(weights_content)):
                        continue

                    for i in self.node_name_mapping:
                        if weight_node.input and not weight_node.input[0].startswith('^') \
                                and weight_node.name in self.node_name_mapping[i].output:
                            continue

                for sub_rule in patterns:
                    if sub_rule[0] != "Dequantize":
                        continue
                    if v != sub_rule[1]:
                        continue

                    if qdq_inserted:
                        if self.node_name_mapping[normal_inputs[0]].node.op != "Dequantize" or \
                           self.node_name_mapping[normal_inputs[1]].node.op != "Dequantize":
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

                        next_node_name = self.node_name_mapping[
                            cur_node_name].output[0]

                        next_node_op = self.node_name_mapping[
                            next_node_name].node.op

                        if next_node_op == sub_rule[-sub_rule_len]:
                            matched_node_name.append(next_node_name)
                            sub_rule_len -= 1
                            cur_node_name = next_node_name
                        else:
                            matched_node_name.clear()
                            self.logger.debug("Fail to match {}.".format(sub_rule))
                            break

                    if sub_rule_len == 1:
                        matched_node_name.append(sub_rule[-1])
                        self.logger.debug("Match {} on nodes {}.".
                                          format(sub_rule, matched_node_name))
                        return sub_rule, matched_node_name

        return None, None
