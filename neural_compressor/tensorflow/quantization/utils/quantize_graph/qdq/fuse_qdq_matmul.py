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
"""Quantize MatMul/BatchMatMul/BatchMatMulV2."""

import numpy as np
from tensorflow.core.framework import graph_pb2, node_def_pb2
from tensorflow.python.framework import dtypes, tensor_util

from neural_compressor.tensorflow.quantization.utils.quantize_graph_common import QuantizeGraphHelper as helper

from ..quantize_graph_base import QuantizeNodeBase


class FuseNodeStartWithMatmul(QuantizeNodeBase):
    """Quantize MatMul/BatchMatMul/BatchMatMulV2 and apply the fusion."""

    exclude_matmul_nodes = []

    def __init__(self, **kwargs):
        """Initialization."""
        super().__init__(**kwargs)

        self.sorted_patterns = sorted(self.patterns, key=lambda i: len(i), reverse=True)
        self.fusion_op_type = set(fusion[1] for fusion in self.patterns)

        self.fusion_mapping = {
            "DequantizeMatMulBiasAddQuantizeV2": self.apply_matmul_biasadd_fusion,
            "DequantizeMatMulQuantizeV2": self.apply_matmul_biasadd_fusion,
            "DequantizeMatMulBiasAddAddQuantizeV2": self.apply_matmul_biasadd_fusion,
            "DequantizeMatMulAddQuantizeV2": self.apply_matmul_biasadd_fusion,
            "DequantizeMatMulBiasAddAddV2QuantizeV2": self.apply_matmul_biasadd_fusion,
            "DequantizeMatMulAddV2QuantizeV2": self.apply_matmul_biasadd_fusion,
            "DequantizeMatMulBiasAddReluQuantizeV2": self.apply_matmul_biasadd_relu_fusion,
            "DequantizeMatMulBiasAddRelu6QuantizeV2": self.apply_matmul_biasadd_relu_fusion,
            "DequantizeMatMulBiasAddLeakyReluQuantizeV2": self.apply_matmul_biasadd_relu_fusion,
            "DequantizeMatMulBiasAddGeluQuantizeV2": self.apply_matmul_biasadd_relu_fusion,
            "DequantizeMatMulBiasAddEluQuantizeV2": self.apply_matmul_biasadd_relu_fusion,
            "DequantizeMatMulBiasAddTanhQuantizeV2": self.apply_matmul_biasadd_relu_fusion,
            "DequantizeMatMulBiasAddSigmoidQuantizeV2": self.apply_matmul_biasadd_relu_fusion,
            "DequantizeMatMulReluQuantizeV2": self.apply_matmul_biasadd_relu_fusion,
            "DequantizeMatMulRelu6QuantizeV2": self.apply_matmul_biasadd_relu_fusion,
            "DequantizeMatMulLeakyReluQuantizeV2": self.apply_matmul_biasadd_relu_fusion,
            "DequantizeMatMulGeluQuantizeV2": self.apply_matmul_biasadd_relu_fusion,
            "DequantizeMatMulEluQuantizeV2": self.apply_matmul_biasadd_relu_fusion,
            "DequantizeMatMulTanhQuantizeV2": self.apply_matmul_biasadd_relu_fusion,
            "DequantizeMatMulSigmoidQuantizeV2": self.apply_matmul_biasadd_relu_fusion,
            "DequantizeBatchMatMulQuantizeV2": self.apply_batchmatmulv2_fusion,
            "DequantizeBatchMatMulV2QuantizeV2": self.apply_batchmatmulv2_fusion,
            "DequantizeBatchMatMulMulQuantizeV2": self.apply_batchmatmulv2_mul_add_fusion,
            "DequantizeBatchMatMulV2MulQuantizeV2": self.apply_batchmatmulv2_mul_add_fusion,
            "DequantizeBatchMatMulAddQuantizeV2": self.apply_batchmatmulv2_mul_add_fusion,
            "DequantizeBatchMatMulV2AddQuantizeV2": self.apply_batchmatmulv2_mul_add_fusion,
            "DequantizeBatchMatMulAddV2QuantizeV2": self.apply_batchmatmulv2_mul_add_fusion,
            "DequantizeBatchMatMulV2AddV2QuantizeV2": self.apply_batchmatmulv2_mul_add_fusion,
            "DequantizeBatchMatMulMulAddV2QuantizeV2": self.apply_batchmatmulv2_mul_add_fusion,
            "DequantizeBatchMatMulV2MulAddV2QuantizeV2": self.apply_batchmatmulv2_mul_add_fusion,
            "DequantizeBatchMatMulMulAddQuantizeV2": self.apply_batchmatmulv2_mul_add_fusion,
            "DequantizeBatchMatMulV2MulAddQuantizeV2": self.apply_batchmatmulv2_mul_add_fusion,
        }

    def apply_matmul_biasadd_relu_fusion(self, match_node_name):
        """Apply dequantize + matmul + biasadd + activation + quantizev2 fusion.

        Dequantize + MatMul + BiasAdd + Relu + QuantizeV2
        Dequantize + MatMul + Relu + QuantizeV2
        Dequantize + MatMul + BiasAdd + Relu6 + QuantizeV2
        Dequantize + MatMul + Relu6 + QuantizeV2
        Dequantize + MatMul + BiasAdd + LeakyRelu + QuantizeV2
        Dequantize + MatMul + LeakyRelu + QuantizeV2
        Dequantize + MatMul + BiasAdd + Gelu + QuantizeV2
        Dequantize + MatMul + Gelu + QuantizeV2
        Dequantize + MatMul + BiasAdd + Elu + QuantizeV2
        Dequantize + MatMul + Elu + QuantizeV2
        Dequantize + MatMul + BiasAdd + Tanh + QuantizeV2
        Dequantize + MatMul + Tanh + QuantizeV2
        Dequantize + MatMul + BiasAdd + Sigmoid + QuantizeV2
        Dequantize + MatMul + Sigmoid + QuantizeV2
        """
        matched_node = self.node_name_mapping[match_node_name[1]]
        control_inputs, normal_inputs = self._get_node_input(matched_node.node.name)

        # QDQ inserted for input0 in phase 1
        _, q_inputs = self._get_node_input(normal_inputs[0])

        # Three scenarios to quantize input1:
        #   a. weight node is const, insert QDQ directly in phase 1
        #   b. weight node is non-const, insert QDQ directly in phase 1
        #   c. weight node is 'Enter' and its parent node is const,
        #      not insert QDQ in phase 1 for easy handling
        weight_name = normal_inputs[1]
        weight_node = self.node_name_mapping[helper.node_name_from_input(weight_name)].node
        enter_node = None
        quantizev2_weights_name = None
        weights_min_name = None
        weights_max_name = None
        # no QDQ inserted for 'Enter' node in phase 1
        if weight_node.op == "Enter":  # pragma: no cover
            parent_node = self.node_name_mapping[helper.node_name_from_input(weight_node.input[0])].node
            # FIXME We only quantize the MatMul op which second input node type is const. This is a
            # workaround for RNN model like LTSM.
            if parent_node.op != "Const":
                self.logger.debug("The weight node of matched_node {} is not Const or Const + Enter, skipped")
                self.exclude_matmul_nodes.append(matched_node.node.name)
                self.output_graph = self.input_graph
                return []
            enter_node = weight_node
            weight_node = parent_node
            weight_name = weight_node.name
        # QDQ inserted for other weight nodes in phase 1
        else:
            _, q_weights_inputs = self._get_node_input(weight_name)
            quantizev2_weights_name = q_weights_inputs[0]

            _, weights_name = self._get_node_input(quantizev2_weights_name)
            weights_min_name = weights_name[1]
            weights_max_name = weights_name[2]
            weight_node = self.node_name_mapping[helper.node_name_from_input(weights_name[0])].node
            weight_name = weight_node.name

        if weight_node.op == "Const":
            weights_content = tensor_util.MakeNdarray(weight_node.attr["value"].tensor)

            if np.any(np.isnan(weights_content)):  # pragma: no cover
                self.exclude_matmul_nodes.append(matched_node.node.name)
                self.output_graph = self.input_graph
                return []

        # If weight node non const, can't insert dummy biasadd to do matmul fusion.
        if weight_node.op != "Const" and len(match_node_name) == 3:
            self.exclude_matmul_nodes.append(matched_node.node.name)
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
            if (add_a_node.op != "Const" and add_b_node.op == "Const") or (
                add_a_node.op != "Const" and add_b_node.op == "Enter"
            ):
                need_insert_dummy_biasadd = 0
                offset = 0
            if need_insert_dummy_biasadd:
                self.apply_matmul_biasadd_fusion(match_node_name[:2] + [match_node_name[-1]])
                return match_node_name[1:2]

        if weight_node.op == "Const":
            q_weights_name, q_weights_min_name, q_weights_max_name = self._intel_cpu_quantize_weight_eightbit(
                matched_node.node.op, self.node_name_mapping[weight_name].node, self.per_channel, enter_node
            )
            if weights_min_name:
                skip_node_name.append(weights_min_name)
            if weights_max_name:
                skip_node_name.append(weights_max_name)
            if quantizev2_weights_name:
                skip_node_name.append(quantizev2_weights_name)
            skip_node_name.append(weight_name)
        else:
            q_weights_name = q_weights_inputs[0]
            q_weights_min_name = q_weights_inputs[1]
            q_weights_max_name = q_weights_inputs[2]

        skip_node_name.append(normal_inputs[0])
        if enter_node:  # pragma: no cover
            skip_node_name.append(enter_node.name)
        else:
            skip_node_name.append(normal_inputs[1])

        for _, node in enumerate(self.input_graph.node):
            if node.name in skip_node_name:
                pass
            elif node.name == match_node_name[1]:
                self.logger.debug("Matched node {} with input {}.".format(node.name, node.input))

                quantized_node_name = node.name + "_eightbit_quantized_mat_mul"
                if need_insert_dummy_biasadd and weight_node.op == "Const":
                    t_b_index = 0 if matched_node.node.attr["transpose_b"].b else 1
                    weights_content = tensor_util.MakeNdarray(weight_node.attr["value"].tensor)
                    bias_size = weights_content.shape[t_b_index]
                    bias_node_name = node.name + "_fake_bias"
                    bias_node = helper.create_constant_node(
                        bias_node_name, [0] * bias_size, dtypes.float32, shape=[bias_size]
                    )

                    if enter_node:  # pragma: no cover
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
                    bias_node_name = self.node_name_mapping[match_node_name[2]].node.input[1]
                relu_node_name = match_node_name[3 - offset]

                all_input_names = q_inputs[:1] + [q_weights_name] + q_inputs[1:]
                all_input_names.append(q_weights_min_name)
                all_input_names.append(q_weights_max_name)
                quantized_node_input_names = (
                    all_input_names[:2] + [bias_node_name] + all_input_names[2:] + control_inputs
                )

                quantized_matmul_node = helper.create_node(
                    "_QuantizedMatMul", quantized_node_name, quantized_node_input_names
                )
                helper.copy_attr(quantized_matmul_node, "transpose_a", node.attr["transpose_a"])
                helper.copy_attr(quantized_matmul_node, "transpose_b", node.attr["transpose_b"])
                helper.set_attr_dtype(quantized_matmul_node, "T1", dtypes.quint8)
                helper.set_attr_dtype(quantized_matmul_node, "T2", dtypes.qint8)
                helper.set_attr_dtype(quantized_matmul_node, "Tout", dtypes.qint32)
                helper.set_attr_string(
                    quantized_matmul_node, "input_quant_mode", b"MIN_FIRST" if self.is_asymmetric else b"SCALED"
                )
                helper.set_attr_string(
                    quantized_matmul_node, "output_quant_mode", b"MIN_FIRST" if self.is_asymmetric else b"SCALED"
                )
                if self.node_name_mapping[relu_node_name].node.op == "Relu":
                    helper.set_attr_string_list(quantized_matmul_node, "fused_ops", [b"BiasAdd", b"Relu"])
                elif self.node_name_mapping[relu_node_name].node.op == "Relu6":
                    helper.set_attr_string_list(quantized_matmul_node, "fused_ops", [b"BiasAdd", b"Relu6"])
                elif self.node_name_mapping[relu_node_name].node.op == "LeakyRelu":
                    helper.set_attr_string_list(quantized_matmul_node, "fused_ops", [b"BiasAdd", b"LeakyRelu"])
                elif self.node_name_mapping[relu_node_name].node.op == "Gelu":
                    if self.node_name_mapping[relu_node_name].node.attr["approximate"].b:
                        helper.set_attr_string_list(
                            quantized_matmul_node, "fused_ops", [b"BiasAdd", b"GeluApproximate"]
                        )
                    else:
                        helper.set_attr_string_list(quantized_matmul_node, "fused_ops", [b"BiasAdd", b"GeluExact"])
                elif self.node_name_mapping[relu_node_name].node.op == "Elu":
                    helper.set_attr_string_list(quantized_matmul_node, "fused_ops", [b"BiasAdd", b"Elu"])
                elif self.node_name_mapping[relu_node_name].node.op == "Tanh":
                    helper.set_attr_string_list(quantized_matmul_node, "fused_ops", [b"BiasAdd", b"Tanh"])
                elif self.node_name_mapping[relu_node_name].node.op == "Sigmoid":
                    helper.set_attr_string_list(quantized_matmul_node, "fused_ops", [b"BiasAdd", b"Sigmoid"])
                helper.set_attr_dtype(quantized_matmul_node, "Tbias", dtypes.float32)
                helper.set_attr_dtype(quantized_matmul_node, "U", dtypes.float32)

                helper.set_attr_type_list(
                    quantized_matmul_node,
                    "Thost_inputs",
                    [
                        dtypes.quint8.as_datatype_enum,
                        dtypes.qint8.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ],
                )
                helper.set_attr_type_list(
                    quantized_matmul_node,
                    "Thost_outputs",
                    [dtypes.qint32.as_datatype_enum, dtypes.float32.as_datatype_enum, dtypes.float32.as_datatype_enum],
                )

                self.add_output_graph_node(quantized_matmul_node)

                quantize_down_name = self._add_quantize_down_nodes(node, quantized_node_name, dtypes.quint8, False)
                self._intel_cpu_add_dequantize_result_node(
                    quantize_down_name, relu_node_name, performance_only=self.performance_only
                )
            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                matmul_node = self.node_name_mapping[match_node_name[1]].node
                matmul_node_output = self.node_name_mapping[match_node_name[1]].output
                if new_node.name in matmul_node_output:
                    for idx, node_input in enumerate(new_node.input):
                        if helper.node_name_from_input(node_input) == matmul_node.name:
                            new_node.input[idx] = node_input.replace(matmul_node.name, quantized_node_name)
                self.add_output_graph_node(new_node)
        return match_node_name

    def apply_matmul_biasadd_fusion(self, match_node_name):
        """Apply dequantize + matmul + biasadd + quantizev2 fusion.

        Dequantize + MatMul + QuantizeV2
        Dequantize + MatMul + BiasAdd + QuantizeV2
        Dequantize + MatMul + Add + QuantizeV2
        Dequantize + MatMul + BiasAdd + Add + QuantizeV2
        Dequantize + MatMul + AddV2 + QuantizeV2
        Dequantize + MatMul + BiasAdd + AddV2 + QuantizeV2
        """
        skip_node_name = match_node_name[2:]
        matched_node = self.node_name_mapping[match_node_name[1]]
        control_inputs, normal_inputs = self._get_node_input(matched_node.node.name)

        # QDQ inserted for input0 in phase 1
        _, q_inputs = self._get_node_input(normal_inputs[0])

        weight_name = normal_inputs[1]
        weight_node = self.node_name_mapping[helper.node_name_from_input(weight_name)].node
        enter_node = None
        weights_min_name = None
        weights_max_name = None
        quantizev2_weights_name = None

        # no QDQ inserted for 'Enter' node in phase 1
        if weight_node.op == "Enter":  # pragma: no cover
            parent_node = self.node_name_mapping[helper.node_name_from_input(weight_node.input[0])].node
            # FIXME We only quantize the MatMul op which second input node type is const. This is a
            # workaround for RNN model like LTSM.
            if parent_node.op != "Const":
                self.logger.debug("The weight node of matched_node {} is not Const or Const + Enter, skipped")
                self.exclude_matmul_nodes.append(matched_node.node.name)
                self.output_graph = self.input_graph
                return []
            enter_node = weight_node
            weight_node = parent_node
            weight_name = weight_node.name
        # QDQ inserted for other weight nodes in phase 1
        else:
            _, q_weights_inputs = self._get_node_input(weight_name)
            quantizev2_weights_name = q_weights_inputs[0]

            _, weights_name = self._get_node_input(quantizev2_weights_name)
            weights_min_name = weights_name[1]
            weights_max_name = weights_name[2]
            weight_node = self.node_name_mapping[helper.node_name_from_input(weights_name[0])].node
            weight_name = weight_node.name

        # TODO Remove below two lines once the TF enabled the QuantizedMatMul while
        # transpose_a could be set to True.
        if matched_node.node.attr["transpose_a"].b is True:  # pragma: no cover
            self.exclude_matmul_nodes.append(matched_node.node.name)
            self.output_graph = self.input_graph
            return []

        if weight_node.op == "Const":
            weights_content = tensor_util.MakeNdarray(weight_node.attr["value"].tensor)

            if np.any(np.isnan(weights_content)):  # pragma: no cover
                self.exclude_matmul_nodes.append(matched_node.node.name)
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

        single_matmul_fusion = True
        if len(match_node_name) == 3:
            if is_shared_output:
                self.output_graph = self.input_graph
                self.exclude_matmul_nodes.append(matched_node.node.name)
                return []
        else:
            second_node = self.node_name_mapping[match_node_name[2]].node
            add_a_node_name = helper.node_name_from_input(second_node.input[0])
            add_a_node = self.node_name_mapping[add_a_node_name].node
            add_b_node_name = helper.node_name_from_input(second_node.input[1])
            add_b_node = self.node_name_mapping[add_b_node_name].node
            if (add_a_node.op != "Const" and add_b_node.op == "Const") or (
                add_a_node.op != "Const" and add_b_node.op == "Enter"
            ):
                single_matmul_fusion = False
            else:
                return self.apply_matmul_biasadd_fusion(match_node_name[:2] + [match_node_name[-1]])

        sum_node_name = ""
        if len(match_node_name) == 4:
            if self.node_name_mapping[match_node_name[2]].node.op in ("Add", "AddV2"):
                sum_index = 1 if match_node_name[1] == self.node_name_mapping[match_node_name[2]].node.input[0] else 0
                sum_node_name = self.node_name_mapping[match_node_name[2]].node.input[sum_index]
                deq_node = self.node_name_mapping[sum_node_name].node
                if deq_node.op != "Dequantize" or deq_node.op.find("Quantize") != -1:
                    return self.apply_matmul_biasadd_fusion(match_node_name[:2] + [match_node_name[-1]])
        if len(match_node_name) == 5:
            if self.node_name_mapping[match_node_name[3]].node.op in ("Add", "AddV2"):
                sum_index = 1 if match_node_name[2] == self.node_name_mapping[match_node_name[3]].node.input[0] else 0
                sum_node_name = self.node_name_mapping[match_node_name[3]].node.input[sum_index]
                deq_node = self.node_name_mapping[sum_node_name].node
                if deq_node.op != "Dequantize" or deq_node.op.find("Quantize") != -1:
                    return self.apply_matmul_biasadd_fusion(match_node_name[:3] + [match_node_name[-1]])

        if weight_node.op == "Const":
            q_weights_name, q_weights_min_name, q_weights_max_name = self._intel_cpu_quantize_weight_eightbit(
                matched_node.node.op, self.node_name_mapping[weight_name].node, self.per_channel, enter_node
            )
            if weights_min_name:
                skip_node_name.append(weights_min_name)
            if weights_max_name:
                skip_node_name.append(weights_max_name)
            if quantizev2_weights_name:
                skip_node_name.append(quantizev2_weights_name)
            skip_node_name.append(weight_name)
        else:
            q_weights_name = q_weights_inputs[0]
            q_weights_min_name = q_weights_inputs[1]
            q_weights_max_name = q_weights_inputs[2]

        skip_node_name.append(normal_inputs[0])
        if enter_node:
            if len(self.node_name_mapping[helper.node_name_from_input(enter_node.name)].output) == 1:
                skip_node_name.append(enter_node.name)
        else:
            skip_node_name.append(normal_inputs[1])

        for _, node in enumerate(self.input_graph.node):
            if node.name in skip_node_name:
                pass
            elif node.name == match_node_name[1]:
                self.logger.debug("Matched node {} with input {}.".format(node.name, node.input))

                quantized_node_name = node.name + "_eightbit_quantized_mat_mul"
                all_input_names = q_inputs[:1] + [q_weights_name] + q_inputs[1:]
                all_input_names.append(q_weights_min_name)
                all_input_names.append(q_weights_max_name)

                if single_matmul_fusion:
                    if sum_node_name:
                        quantized_node_input_names = (
                            all_input_names[:2] + [sum_node_name] + all_input_names[2:] + control_inputs
                        )
                    else:
                        quantized_node_input_names = all_input_names[:2] + all_input_names[2:] + control_inputs
                else:
                    bias_node_name = self.node_name_mapping[match_node_name[2]].node.input[1]
                    if sum_node_name:
                        quantized_node_input_names = (
                            all_input_names[:2]
                            + [bias_node_name]
                            + [sum_node_name]
                            + all_input_names[2:]
                            + control_inputs
                        )
                    else:
                        quantized_node_input_names = (
                            all_input_names[:2] + [bias_node_name] + all_input_names[2:] + control_inputs
                        )

                quantized_matmul_node = helper.create_node(
                    "_QuantizedMatMul", quantized_node_name, quantized_node_input_names
                )
                helper.copy_attr(quantized_matmul_node, "transpose_a", node.attr["transpose_a"])
                helper.copy_attr(quantized_matmul_node, "transpose_b", node.attr["transpose_b"])
                helper.set_attr_dtype(quantized_matmul_node, "T1", dtypes.quint8)
                helper.set_attr_dtype(quantized_matmul_node, "T2", dtypes.qint8)
                helper.set_attr_dtype(quantized_matmul_node, "Tout", dtypes.qint32)
                helper.set_attr_dtype(quantized_matmul_node, "U", dtypes.float32)
                helper.set_attr_string(
                    quantized_matmul_node, "input_quant_mode", b"MIN_FIRST" if self.is_asymmetric else b"SCALED"
                )
                helper.set_attr_string(
                    quantized_matmul_node, "output_quant_mode", b"MIN_FIRST" if self.is_asymmetric else b"SCALED"
                )
                helper.set_attr_dtype(quantized_matmul_node, "Tbias", dtypes.float32)
                if sum_node_name:
                    helper.set_attr_string_list(quantized_matmul_node, "fused_ops", [b"BiasAdd", b"Add"])
                    helper.set_attr_type_list(
                        quantized_matmul_node,
                        "Thost_inputs",
                        [
                            dtypes.quint8.as_datatype_enum,
                            dtypes.qint8.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                        ],
                    )
                else:
                    if not single_matmul_fusion:
                        helper.set_attr_string_list(quantized_matmul_node, "fused_ops", [b"BiasAdd"])
                        helper.set_attr_type_list(
                            quantized_matmul_node,
                            "Thost_inputs",
                            [
                                dtypes.quint8.as_datatype_enum,
                                dtypes.qint8.as_datatype_enum,
                                dtypes.float32.as_datatype_enum,
                                dtypes.float32.as_datatype_enum,
                                dtypes.float32.as_datatype_enum,
                                dtypes.float32.as_datatype_enum,
                                dtypes.float32.as_datatype_enum,
                            ],
                        )
                    else:
                        helper.set_attr_type_list(
                            quantized_matmul_node,
                            "Thost_inputs",
                            [
                                dtypes.quint8.as_datatype_enum,
                                dtypes.qint8.as_datatype_enum,
                                dtypes.float32.as_datatype_enum,
                                dtypes.float32.as_datatype_enum,
                                dtypes.float32.as_datatype_enum,
                                dtypes.float32.as_datatype_enum,
                            ],
                        )
                helper.set_attr_type_list(
                    quantized_matmul_node,
                    "Thost_outputs",
                    [dtypes.qint32.as_datatype_enum, dtypes.float32.as_datatype_enum, dtypes.float32.as_datatype_enum],
                )

                self.add_output_graph_node(quantized_matmul_node)
                requantize_type = dtypes.qint8

                quantize_down_name = self._add_quantize_down_nodes(node, quantized_node_name, requantize_type, False)
                if sum_node_name:
                    self._intel_cpu_add_dequantize_result_node(
                        quantize_down_name,
                        match_node_name[2] if single_matmul_fusion else match_node_name[3],
                        requantize_type,
                        performance_only=self.performance_only,
                    )
                else:
                    self._intel_cpu_add_dequantize_result_node(
                        quantize_down_name,
                        match_node_name[1] if single_matmul_fusion else match_node_name[2],
                        requantize_type,
                        performance_only=self.performance_only,
                    )
            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)
        return match_node_name

    def apply_batchmatmulv2_fusion(self, match_node_name):  # pragma: no cover
        """Apply dequantize + batchmatmul/batchmatmulv2 + quantizev2 fusion.

        Dequantize + BatchMatMulV2 + QuantizeV2
        """
        skip_node_name = match_node_name[2:]
        matched_node = self.node_name_mapping[match_node_name[1]]
        control_inputs, normal_inputs = self._get_node_input(matched_node.node.name)

        _, q_x_inputs = self._get_node_input(normal_inputs[0])
        quantizev2_input_name = q_x_inputs[0]
        weight_name = normal_inputs[1]
        weight_node = self.node_name_mapping[helper.node_name_from_input(weight_name)].node
        enter_node = None
        weights_min_name = None
        weights_max_name = None
        quantizev2_y_name = None

        if weight_node.op == "Enter":
            parent_node = self.node_name_mapping[helper.node_name_from_input(weight_node.input[0])].node
            # FIXME We only quantize the MatMul op which second input node type is const. This is a
            # workaround for RNN model like LTSM.
            if parent_node.op != "Const":
                self.logger.debug("The weight node of matched_node {} is not Const or Const + Enter, skipped")
                self.exclude_matmul_nodes.append(matched_node.node.name)
                self.output_graph = self.input_graph
                return []
            enter_node = weight_node
            weight_node = parent_node
            weight_name = weight_node.name
        else:
            _, q_y_inputs = self._get_node_input(normal_inputs[1])
            quantizev2_y_name = q_y_inputs[0]
            _, weights_name = self._get_node_input(quantizev2_y_name)
            weights_min_name = weights_name[1]
            weights_max_name = weights_name[2]
            weight_node = self.node_name_mapping[helper.node_name_from_input(weights_name[0])].node
            weight_name = weight_node.name

        if weight_node.op == "Const":
            weights_content = tensor_util.MakeNdarray(weight_node.attr["value"].tensor)

            if np.any(np.isnan(weights_content)):
                self.output_graph = self.input_graph
                self.exclude_matmul_nodes.append(matched_node.node.name)
                return []

            for i in self.node_name_mapping:
                if (
                    weight_node.input
                    and not weight_node.input[0].startswith("^")
                    and weight_node.name in self.node_name_mapping[i].output
                ):
                    self.output_graph = self.input_graph
                    self.exclude_matmul_nodes.append(matched_node.node.name)
                    return []

            q_weights_name, q_weights_min_name, q_weights_max_name = self._intel_cpu_quantize_weight_eightbit(
                matched_node.node.op, self.node_name_mapping[weight_name].node, self.per_channel, enter_node
            )

            if weights_min_name:
                skip_node_name.append(weights_min_name)
            if weights_max_name:
                skip_node_name.append(weights_max_name)
            if quantizev2_y_name:
                skip_node_name.append(quantizev2_y_name)
            skip_node_name.append(weight_name)
        else:
            q_weights_name = q_y_inputs[0]
            q_weights_min_name = q_y_inputs[1]
            q_weights_max_name = q_y_inputs[2]

        skip_node_name.append(normal_inputs[0])
        if enter_node:
            skip_node_name.append(enter_node.name)
        else:
            skip_node_name.append(normal_inputs[1])

        batchmatmul_next_node = {}
        for _, node in enumerate(self.input_graph.node):
            quantized_node_name = ""
            if node.name in skip_node_name:
                pass
            elif node.name == match_node_name[1]:
                self.logger.debug("Matched node {} with input {}.".format(node.name, node.input))

                quantized_node_name = node.name + "_eightbit_quantized_batch_matmul_v2"
                all_input_names = [quantizev2_input_name] + [q_weights_name] + q_x_inputs[1:]
                all_input_names.append(q_weights_min_name)
                all_input_names.append(q_weights_max_name)

                quantized_node_input_names = all_input_names + control_inputs

                quantized_matmul_node = helper.create_node(
                    "_QuantizedBatchMatMul", quantized_node_name, quantized_node_input_names
                )

                helper.copy_attr(quantized_matmul_node, "adj_x", node.attr["adj_x"])
                helper.copy_attr(quantized_matmul_node, "adj_y", node.attr["adj_y"])
                helper.set_attr_dtype(quantized_matmul_node, "T1", dtypes.qint8)
                helper.set_attr_dtype(quantized_matmul_node, "T2", dtypes.qint8)
                helper.set_attr_dtype(quantized_matmul_node, "U", dtypes.float32)
                helper.set_attr_dtype(quantized_matmul_node, "Tout", dtypes.float32)
                helper.set_attr_string(quantized_matmul_node, "input_quant_mode", b"SCALED")
                helper.set_attr_string(quantized_matmul_node, "output_quant_mode", b"SCALED")
                helper.set_attr_string_list(quantized_matmul_node, "fused_ops", [b"Dequantize"])
                helper.set_attr_type_list(
                    quantized_matmul_node,
                    "Thost_inputs",
                    [
                        dtypes.qint8.as_datatype_enum,
                        dtypes.qint8.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ],
                )
                helper.set_attr_type_list(quantized_matmul_node, "Thost_outputs", [dtypes.float32.as_datatype_enum])

                self.add_output_graph_node(quantized_matmul_node)
                for i in self.node_name_mapping[node.name].output:
                    batchmatmul_next_node[i] = (quantized_node_name, node.name)
            else:
                new_node = node_def_pb2.NodeDef()
                if batchmatmul_next_node.get(node.name):
                    for index, name in enumerate(node.input):
                        if name == batchmatmul_next_node[node.name][1]:
                            node.input[index] = batchmatmul_next_node[node.name][0]
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)
        return match_node_name

    def apply_batchmatmulv2_mul_add_fusion(self, match_node_name):  # pragma: no cover
        """Apply dequantize + batchmatmul/batchmatmulv2 + mul + add + quantizev2 fusion.

        Dequantize + BatchMatMulV2 + Mul + QuantizeV2
        Dequantize + BatchMatMulV2 + Add + QuantizeV2
        Dequantize + BatchMatMulV2 + AddV2 + QuantizeV2
        Dequantize + BatchMatMulV2 + Mul + Add + QuantizeV2
        Dequantize + BatchMatMulV2 + Mul + AddV2 + QuantizeV2
        """
        skip_node_name = match_node_name[2:]
        matched_node = self.node_name_mapping[match_node_name[1]]
        # oneDNN limitation: add tensor ndim must be 4
        if len(match_node_name) == 4 and self.node_name_mapping[match_node_name[2]].node.op in ("Add", "AddV2"):
            add_node_input_name = self.node_name_mapping[match_node_name[2]].node.input[1]
            if add_node_input_name == matched_node.node.name:
                add_node_input_name = self.node_name_mapping[match_node_name[2]].node.input[0]
            add_input_node = self.node_name_mapping[add_node_input_name].node
            if add_input_node.op != "Const":
                return self.apply_batchmatmulv2_fusion(match_node_name[:2] + [match_node_name[-1]])

            shape = tensor_util.MakeNdarray(add_input_node.attr["value"].tensor)
            if shape.ndim != 4:
                return self.apply_batchmatmulv2_fusion(match_node_name[:2] + [match_node_name[-1]])

        control_inputs, normal_inputs = self._get_node_input(matched_node.node.name)

        weight_name = normal_inputs[1]
        weight_node = self.node_name_mapping[helper.node_name_from_input(weight_name)].node
        _, q_x_inputs = self._get_node_input(normal_inputs[0])
        enter_node = None
        weights_min_name = None
        weights_max_name = None
        quantizev2_weights_name = None
        if weight_node.op == "Enter":
            parent_node = self.node_name_mapping[helper.node_name_from_input(weight_node.input[0])].node
            # FIXME We only quantize the MatMul op which second input node type is const. This is a
            # workaround for RNN model like LTSM.
            if parent_node.op != "Const":
                self.logger.debug("The weight node of matched_node {} is not Const or Const + Enter, skipped")
                self.exclude_matmul_nodes.append(matched_node.node.name)
                self.output_graph = self.input_graph
                return []
            enter_node = weight_node
            weight_node = parent_node
            weight_name = weight_node.name
        else:
            _, q_weights_inputs = self._get_node_input(normal_inputs[1])
            quantizev2_weights_name = q_weights_inputs[0]
            _, weights_name = self._get_node_input(quantizev2_weights_name)
            weights_min_name = weights_name[1]
            weights_max_name = weights_name[2]
            weight_node = self.node_name_mapping[helper.node_name_from_input(weights_name[0])].node
            weight_name = weight_node.name

        if weight_node.op == "Const":
            weights_content = tensor_util.MakeNdarray(weight_node.attr["value"].tensor)

            if np.any(np.isnan(weights_content)):
                self.exclude_matmul_nodes.append(matched_node.node.name)
                self.output_graph = self.input_graph
                return []

            for i in self.node_name_mapping:
                if (
                    weight_node.input
                    and not weight_node.input[0].startswith("^")
                    and weight_node.name in self.node_name_mapping[i].output
                ):
                    self.output_graph = self.input_graph
                    self.exclude_matmul_nodes.append(matched_node.node.name)
                    return []

            q_weights_name, q_weights_min_name, q_weights_max_name = self._intel_cpu_quantize_weight_eightbit(
                matched_node.node.op, self.node_name_mapping[weight_name].node, self.per_channel, enter_node
            )

            if weights_min_name:
                skip_node_name.append(weights_min_name)
            if weights_max_name:
                skip_node_name.append(weights_max_name)
            if quantizev2_weights_name:
                skip_node_name.append(quantizev2_weights_name)
            skip_node_name.append(weight_name)
        else:
            q_weights_name = q_weights_inputs[0]
            q_weights_min_name = q_weights_inputs[1]
            q_weights_max_name = q_weights_inputs[2]

        skip_node_name.append(normal_inputs[0])
        if enter_node:
            skip_node_name.append(enter_node.name)
        else:
            skip_node_name.append(normal_inputs[1])

        batchmatmul_next_node = {}
        for _, node in enumerate(self.input_graph.node):
            if node.name in skip_node_name:
                pass
            elif node.name == match_node_name[1]:
                self.logger.debug("Matched node {} with input {}.".format(node.name, node.input))
                quantized_node_name = node.name + "_eightbit_quantized_batch_matmul_v2"

                if len(match_node_name) == 4:
                    if self.node_name_mapping[match_node_name[2]].node.op == "Mul":
                        mul_node_name = self.node_name_mapping[match_node_name[2]].node.input[1]
                        all_input_names = q_x_inputs[:1] + [q_weights_name] + [mul_node_name] + q_x_inputs[1:]
                        all_input_names.append(q_weights_min_name)
                        all_input_names.append(q_weights_max_name)
                    else:
                        add_node_name = self.node_name_mapping[match_node_name[2]].node.input[1]
                        all_input_names = q_x_inputs[:1] + [q_weights_name] + [add_node_name] + q_x_inputs[1:]
                        all_input_names.append(q_weights_min_name)
                        all_input_names.append(q_weights_max_name)
                    skip_node_name.append(match_node_name[2])
                else:
                    mul_node_name = self.node_name_mapping[match_node_name[2]].node.input[1]
                    add_node_name = self.node_name_mapping[match_node_name[3]].node.input[1]
                    skip_node_name.append(match_node_name[2])
                    skip_node_name.append(match_node_name[3])
                    all_input_names = (
                        q_x_inputs[:1] + [q_weights_name] + [mul_node_name] + [add_node_name] + q_x_inputs[1:]
                    )
                    all_input_names.append(q_weights_min_name)
                    all_input_names.append(q_weights_max_name)

                quantized_node_input_names = all_input_names + control_inputs

                quantized_matmul_node = helper.create_node(
                    "_QuantizedBatchMatMul", quantized_node_name, quantized_node_input_names
                )

                helper.copy_attr(quantized_matmul_node, "adj_x", node.attr["adj_x"])
                helper.copy_attr(quantized_matmul_node, "adj_y", node.attr["adj_y"])
                helper.set_attr_dtype(quantized_matmul_node, "T1", dtypes.qint8)
                helper.set_attr_dtype(quantized_matmul_node, "T2", dtypes.qint8)
                helper.set_attr_dtype(quantized_matmul_node, "U", dtypes.float32)
                helper.set_attr_dtype(quantized_matmul_node, "Tout", dtypes.float32)
                helper.set_attr_string(quantized_matmul_node, "input_quant_mode", b"SCALED")
                helper.set_attr_string(quantized_matmul_node, "output_quant_mode", b"SCALED")
                if len(match_node_name) == 4:
                    if self.node_name_mapping[match_node_name[2]].node.op == "Mul":
                        helper.set_attr_string_list(quantized_matmul_node, "fused_ops", [b"Mul", b"Dequantize"])
                    else:
                        helper.set_attr_string_list(quantized_matmul_node, "fused_ops", [b"Add", b"Dequantize"])
                    helper.set_attr_type_list(
                        quantized_matmul_node,
                        "Thost_inputs",
                        [
                            dtypes.qint8.as_datatype_enum,
                            dtypes.qint8.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                        ],
                    )
                else:
                    helper.set_attr_string_list(quantized_matmul_node, "fused_ops", [b"Mul", b"Add", b"Dequantize"])
                    helper.set_attr_type_list(
                        quantized_matmul_node,
                        "Thost_inputs",
                        [
                            dtypes.qint8.as_datatype_enum,
                            dtypes.qint8.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                            dtypes.float32.as_datatype_enum,
                        ],
                    )
                helper.set_attr_type_list(quantized_matmul_node, "Thost_outputs", [dtypes.float32.as_datatype_enum])

                self.add_output_graph_node(quantized_matmul_node)
                attr_fused_ops = "".join(
                    x
                    for x in quantized_matmul_node.attr["fused_ops"]
                    .SerializeToString()
                    .decode("UTF-8", "ignore")
                    .strip()
                    if x.isprintable()
                )
                if "MulAdd" in attr_fused_ops:
                    for i in self.node_name_mapping[match_node_name[3]].output:
                        batchmatmul_next_node[i] = (quantized_node_name, match_node_name[3])
                else:
                    for i in self.node_name_mapping[match_node_name[2]].output:
                        batchmatmul_next_node[i] = (quantized_node_name, match_node_name[2])
            else:
                new_node = node_def_pb2.NodeDef()
                if batchmatmul_next_node.get(node.name):
                    for index, name in enumerate(node.input):
                        if name == batchmatmul_next_node[node.name][1]:
                            node.input[index] = batchmatmul_next_node[node.name][0]
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)
        return match_node_name

    def get_longest_fuse(self):
        """Get the longest fusion pattern."""
        self._get_op_list()
        matched_rule, matched_node_name = self._is_match_matmul(self.sorted_patterns)
        return matched_rule, matched_node_name

    def apply_the_transform(self):
        """Quantize MatMul/BatchMatMul/BatchMatMulV2 and apply the fusion pattern."""
        self._get_op_list()
        matched_rule, matched_node_name = self._is_match_matmul(self.sorted_patterns, True)
        if matched_node_name:
            _, normal_inputs = self._get_node_input(matched_node_name[1])
        if matched_node_name and self.node_name_mapping[normal_inputs[0]].node.op == matched_node_name[0]:
            self.output_graph = graph_pb2.GraphDef()
            fusion_name = "".join(matched_rule)
            if fusion_name in self.fusion_mapping:
                _ = self.fusion_mapping[fusion_name](matched_node_name)
            else:  # pragma: no cover
                self.logger.debug("Unknown fusion pattern {}.".format(fusion_name))
                if self.remove_redundant_quant_flag:
                    self.input_graph = self.remove_redundant_quantization(self.input_graph)
                return self.input_graph, self.exclude_matmul_nodes

            self.input_graph = self.output_graph
            self._reset_output_node_maps()
            if self.remove_redundant_quant_flag:
                self.output_graph = self.remove_redundant_quantization(self.output_graph)
            return self.output_graph, self.exclude_matmul_nodes

        if self.remove_redundant_quant_flag:
            self.input_graph = self.remove_redundant_quantization(self.input_graph)
        return self.input_graph, self.exclude_matmul_nodes

    def _is_match_matmul(self, patterns, qdq_inserted=False):
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

                # Disable BatchMatMul quantization temporarily for its bad performance
                # Enable it again when the performance issue is fixed.
                if cur_node.op in ("BatchMatMulV2", "BatchMatMul"):
                    self.exclude_matmul_nodes.append(cur_node.name)
                    continue

                control_inputs, normal_inputs = self._get_node_input(cur_node.name)
                weight_name = normal_inputs[1]
                weight_node = self.node_name_mapping[helper.node_name_from_input(weight_name)].node
                if not qdq_inserted:
                    # FIXME We only quantize the MatMul op which second input node type is const.
                    # This is a workaround for RNN model like LTSM.
                    parent_node = None
                    if cur_node.op == "MatMul" and not self.itex_mode:
                        if control_inputs:  # pragma: no cover
                            self.exclude_matmul_nodes.append(cur_node.name)
                            continue
                        if weight_node.op != "Const":
                            if weight_node.input:
                                parent_node = self.node_name_mapping[
                                    helper.node_name_from_input(weight_node.input[0])
                                ].node
                                if weight_node.op == "Enter":  # pragma: no cover
                                    if len(self.node_name_mapping[helper.node_name_from_input(weight_name)].output) > 1:
                                        self.exclude_matmul_nodes.append(cur_node.name)
                                        continue
                                    if parent_node.op == "Const":
                                        weight_node = parent_node
                                        weights_content = tensor_util.MakeNdarray(weight_node.attr["value"].tensor)
                                        if np.any(np.isnan(weights_content)):
                                            self.exclude_matmul_nodes.append(cur_node.name)
                                            continue
                                    else:
                                        self.exclude_matmul_nodes.append(cur_node.name)
                                        continue
                        else:
                            weights_content = tensor_util.MakeNdarray(weight_node.attr["value"].tensor)
                            if np.any(np.isnan(weights_content)):  # pragma: no cover
                                self.exclude_matmul_nodes.append(cur_node.name)
                                continue

                        # TODO Remove below two lines once the TF enabled the QuantizedMatMul while
                        # transpose_a could be set to True.
                        if cur_node.attr["transpose_a"].b is True:  # pragma: no cover
                            self.exclude_matmul_nodes.append(cur_node.name)
                            continue

                for sub_rule in patterns:
                    if sub_rule[0] != "Dequantize":
                        self.exclude_matmul_nodes.append(cur_node.name)
                        continue
                    if v != sub_rule[1]:
                        self.exclude_matmul_nodes.append(cur_node.name)
                        continue

                    if qdq_inserted:
                        if control_inputs:  # pragma: no cover
                            self.exclude_matmul_nodes.append(cur_node.name)
                            continue
                        if self.node_name_mapping[normal_inputs[0]].node.op != "Dequantize" or self.node_name_mapping[
                            normal_inputs[1]
                        ].node.op not in ("Dequantize", "Enter"):
                            self.exclude_matmul_nodes.append(cur_node.name)
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

                        next_node_op = self.node_name_mapping[next_node_name].node.op

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
                        self.logger.debug("Match {} on nodes {}.".format(sub_rule, matched_node_name))
                        return sub_rule, matched_node_name

        return None, None
