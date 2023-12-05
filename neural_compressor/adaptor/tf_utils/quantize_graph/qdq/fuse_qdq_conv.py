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
"""Quantize Conv2D/Conv3D/DepthwiseConv2dNative."""

import numpy as np
import tensorflow as tf
from tensorflow.core.framework import graph_pb2, node_def_pb2
from tensorflow.python.framework import dtypes, tensor_util

from neural_compressor.adaptor.tf_utils.quantize_graph_common import QuantizeGraphHelper as helper

from ..quantize_graph_base import QuantizeNodeBase


class FuseNodeStartWithConv2d(QuantizeNodeBase):
    """Quantize Conv2D/Conv3D/DepthwiseConv2dNative to int8 op."""

    exclude_conv_nodes = []

    def __init__(self, **kwargs):
        """Initialization."""
        super().__init__(**kwargs)
        self.sorted_patterns = sorted(self.patterns, key=lambda i: len(i), reverse=True)
        if self.new_api:
            # fmt: off
            self.fusion_mapping = {
                'DequantizeConv2DBiasAddQuantizeV2': self.apply_newly_conv_biasadd_fusion,
                'DequantizeConv2DBiasAddAddNReluQuantizeV2': self.apply_newly_conv_biasadd_addn_relu_fusion,
                'DequantizeConv2DAddNReluQuantizeV2': self.apply_newly_conv_biasadd_relu_fusion,
                'DequantizeConv2DBiasAddAddNRelu6QuantizeV2': self.apply_newly_conv_biasadd_addn_relu_fusion,
                'DequantizeConv2DAddNRelu6QuantizeV2': self.apply_newly_conv_biasadd_relu_fusion,
                'DequantizeConv2DBiasAddAddV2ReluQuantizeV2': self.apply_newly_conv_biasadd_addn_relu_fusion,
                'DequantizeConv2DAddV2ReluQuantizeV2': self.apply_newly_conv_biasadd_relu_fusion,
                'DequantizeConv2DBiasAddAddV2Relu6QuantizeV2': self.apply_newly_conv_biasadd_addn_relu_fusion,
                'DequantizeConv2DAddV2Relu6QuantizeV2': self.apply_newly_conv_biasadd_relu_fusion,
                'DequantizeConv2DBiasAddAddReluQuantizeV2': self.apply_newly_conv_biasadd_addn_relu_fusion,
                'DequantizeConv2DBiasAddRelu6QuantizeV2': self.apply_newly_conv_biasadd_relu_fusion,
                'DequantizeConv2DRelu6QuantizeV2': self.apply_newly_conv_biasadd_relu_fusion,
                'DequantizeConv2DBiasAddReluQuantizeV2': self.apply_newly_conv_biasadd_relu_fusion,
                'DequantizeConv2DReluQuantizeV2': self.apply_newly_conv_biasadd_relu_fusion,
                'DequantizeConv2DBiasAddEluQuantizeV2': self.apply_newly_conv_biasadd_relu_fusion,
                'DequantizeConv2DEluQuantizeV2': self.apply_newly_conv_biasadd_relu_fusion,
                'DequantizeConv2DBiasAddLeakyReluQuantizeV2': self.apply_newly_conv_biasadd_relu_fusion,
                'DequantizeConv2DLeakyReluQuantizeV2': self.apply_newly_conv_biasadd_relu_fusion,
                'DequantizeConv2DBiasAddSigmoidQuantizeV2': self.apply_newly_conv_biasadd_relu_fusion,
                'DequantizeConv2DSigmoidQuantizeV2': self.apply_newly_conv_biasadd_relu_fusion,
                'DequantizeConv2DBiasAddLeakyReluAddV2QuantizeV2': self.apply_newly_conv_biasadd_addn_relu_fusion,
                'DequantizeConv2DBiasAddLeakyReluAddQuantizeV2': self.apply_newly_conv_biasadd_addn_relu_fusion,
                'DequantizeConv2DBiasAddReluAddV2QuantizeV2': self.apply_newly_conv_biasadd_addn_relu_fusion,
                'DequantizeConv2DBiasAddReluAddQuantizeV2': self.apply_newly_conv_biasadd_addn_relu_fusion,
                'DequantizeConv2DBiasAddAddLeakyReluQuantizeV2': self.apply_newly_conv_biasadd_addn_relu_fusion,
                'DequantizeConv2DBiasAddAddV2LeakyReluQuantizeV2': self.apply_newly_conv_biasadd_addn_relu_fusion,
                'DequantizeConv2DAddLeakyReluQuantizeV2': self.apply_newly_conv_biasadd_addn_relu_fusion,
                'DequantizeConv2DAddV2LeakyReluQuantizeV2': self.apply_newly_conv_biasadd_addn_relu_fusion,
                'DequantizeConv2DLeakyReluAddV2QuantizeV2': self.apply_newly_conv_biasadd_addn_relu_fusion,
                'DequantizeConv2DLeakyReluAddQuantizeV2': self.apply_newly_conv_biasadd_addn_relu_fusion,
                'DequantizeConv2DReluAddV2QuantizeV2': self.apply_newly_conv_biasadd_addn_relu_fusion,
                'DequantizeConv2DReluAddQuantizeV2': self.apply_newly_conv_biasadd_addn_relu_fusion,
                'DequantizeConv2DAddRelu6QuantizeV2': self.apply_newly_conv_biasadd_relu_fusion,
                'DequantizeConv2DAddReluQuantizeV2': self.apply_newly_conv_biasadd_relu_fusion,
                'DequantizeConv2DBiasAddAddRelu6MulMulQuantizeV2': self.apply_conv_biasadd_hardswish_fusion,
                'DequantizeConv2DAddRelu6MulMulQuantizeV2': self.apply_conv_biasadd_hardswish_fusion,
                'DequantizeConv2DBiasAddswish_f32QuantizeV2': self.apply_newly_conv_biasadd_swishf32_fusion,
                'DequantizeConv2DAddswish_f32QuantizeV2': self.apply_newly_conv_biasadd_swishf32_fusion,
                'DequantizeConv2DAddV2swish_f32QuantizeV2': self.apply_newly_conv_biasadd_swishf32_fusion,
                'DequantizeConv2Dswish_f32QuantizeV2': self.apply_newly_conv_biasadd_swishf32_fusion,
                'DequantizeDepthwiseConv2dNativeBiasAddAddRelu6MulMulQuantizeV2':
                               self.apply_conv_biasadd_hardswish_fusion,
                'DequantizeDepthwiseConv2dNativeAddRelu6MulMulQuantizeV2': self.apply_conv_biasadd_hardswish_fusion,
                'DequantizeDepthwiseConv2dNativeBiasAddswish_f32QuantizeV2':
                               self.apply_newly_conv_biasadd_swishf32_fusion,
                'DequantizeDepthwiseConv2dNativeAddswish_f32QuantizeV2': self.apply_newly_conv_biasadd_swishf32_fusion,
                'DequantizeDepthwiseConv2dNativeAddV2swish_f32QuantizeV2':
                               self.apply_newly_conv_biasadd_swishf32_fusion,
                'DequantizeDepthwiseConv2dNativeswish_f32QuantizeV2': self.apply_newly_conv_biasadd_swishf32_fusion,
                'DequantizeDepthwiseConv2dNativeAddRelu6QuantizeV2': self.apply_newly_conv_biasadd_relu_fusion,
                'DequantizeDepthwiseConv2dNativeBiasAddReluQuantizeV2': self.apply_newly_conv_biasadd_relu_fusion,
                'DequantizeDepthwiseConv2dNativeReluQuantizeV2': self.apply_newly_conv_biasadd_relu_fusion,
                'DequantizeDepthwiseConv2dNativeRelu6QuantizeV2': self.apply_newly_conv_biasadd_relu_fusion,
                'DequantizeDepthwiseConv2dNativeBiasAddQuantizeV2': self.apply_newly_conv_biasadd_fusion,
                'DequantizeDepthwiseConv2dNativeBiasAddLeakyReluQuantizeV2':
                               self.apply_newly_conv_biasadd_relu_fusion,
                'DequantizeDepthwiseConv2dNativeLeakyReluQuantizeV2': self.apply_newly_conv_biasadd_relu_fusion,
                'DequantizeDepthwiseConv2dNativeBiasAddRelu6QuantizeV2': self.apply_newly_conv_biasadd_relu_fusion,
                'DequantizeConv2DQuantizeV2': self.apply_newly_conv_single_fusion,
                'DequantizeConv2DAddQuantizeV2': self.apply_newly_conv_biasadd_fusion,
                'DequantizeConv2DAddV2QuantizeV2': self.apply_newly_conv_biasadd_fusion,
                'DequantizeConv2DAddAddQuantizeV2': self.apply_newly_conv_biasadd_addn_fusion,
                'DequantizeConv2DAddV2AddQuantizeV2': self.apply_newly_conv_biasadd_addn_fusion,
                'DequantizeConv2DBiasAddAddQuantizeV2': self.apply_newly_conv_biasadd_addn_fusion,
                'DequantizeConv2DAddAddReluQuantizeV2': self.apply_newly_conv_biasadd_addn_relu_fusion,
                'DequantizeConv3DQuantizeV2': self.apply_conv3d_single_fusion,
                'DequantizeConv3DBiasAddQuantizeV2': self.apply_conv3d_add_fusion,
                'DequantizeConv3DBiasAddAddQuantizeV2': self.apply_conv3d_add_addn_fusion,
                'DequantizeConv3DBiasAddAddV2QuantizeV2': self.apply_conv3d_add_addn_fusion,
                'DequantizeConv3DAddV2AddV2QuantizeV2': self.apply_conv3d_add_addn_fusion,
                'DequantizeConv3DAddV2AddV2ReluQuantizeV2': self.apply_conv3d_add_addn_relu_fusion,
                'DequantizeConv3DBiasAddAddV2ReluQuantizeV2': self.apply_conv3d_add_addn_relu_fusion,
                'DequantizeConv3DAddQuantizeV2': self.apply_conv3d_add_fusion,
                'DequantizeConv3DAddReluQuantizeV2': self.apply_conv3d_add_relu_fusion,
                'DequantizeConv3DAddV2QuantizeV2': self.apply_conv3d_add_fusion,
                'DequantizeConv3DAddV2ReluQuantizeV2': self.apply_conv3d_add_relu_fusion,
                'DequantizeConv3DReluQuantizeV2': self.apply_conv3d_add_relu_fusion,
                'DequantizeConv3DBiasAddReluQuantizeV2': self.apply_conv3d_add_relu_fusion,
                'DequantizeConv3DRelu6QuantizeV2': self.apply_conv3d_add_relu_fusion,
                'DequantizeConv3DBiasAddRelu6QuantizeV2': self.apply_conv3d_add_relu_fusion,
                'DequantizeConv3DAddRelu6QuantizeV2': self.apply_conv3d_add_relu_fusion,
                'DequantizeConv3DEluQuantizeV2': self.apply_conv3d_add_relu_fusion,
                'DequantizeConv3DBiasAddEluQuantizeV2': self.apply_conv3d_add_relu_fusion,
                'DequantizeConv3DAddEluQuantizeV2': self.apply_conv3d_add_relu_fusion,
                'DequantizeConv3DLeakyReluQuantizeV2': self.apply_conv3d_add_relu_fusion,
                'DequantizeConv3DBiasAddLeakyReluQuantizeV2': self.apply_conv3d_add_relu_fusion,
                'DequantizeConv3DAddLeakyReluQuantizeV2': self.apply_conv3d_add_relu_fusion,
                'DequantizeDepthwiseConv2dNativeQuantizeV2': self.apply_newly_conv_single_fusion
            }
            # fmt: on

    def _insert_dummy_biasadd(self, match_node_name, matched_node):
        """Insert dummy biasadd for fusion."""
        target_node_name = matched_node.node.name
        op_a_node_name = helper.node_name_from_input(matched_node.node.input[0])
        op_a_node = self.node_name_mapping[op_a_node_name].node

        _, normal_inputs = self._get_node_input(matched_node.node.name)
        _, q_weights_inputs = self._get_node_input(normal_inputs[1])

        quantizev2_weights_name = q_weights_inputs[0]
        _, weights_name = self._get_node_input(quantizev2_weights_name)
        op_b_node_name = weights_name[0]
        op_b_node = self.node_name_mapping[op_b_node_name].node

        if op_a_node.op == "Const" and op_b_node.op != "Const":
            pass
        else:
            from neural_compressor.adaptor.tf_utils.graph_util import GraphAnalyzer

            g = GraphAnalyzer()
            g.graph = self.input_graph
            graph_info = g.parse_graph()
            next_node_names = graph_info[matched_node.node.name].outputs
            bias_node_name = target_node_name + "_dummy_biasadd"
            bias_const_node_name = target_node_name + "_fake_const"

            if (
                matched_node.node.op in ("Conv2D" or "DepthwiseConv2dNative")
                and matched_node.node.attr["data_format"].s == b"NHWC"
            ):
                t_b_index = 3
            elif (
                matched_node.node.op in ("Conv2D" or "DepthwiseConv2dNative")
                and matched_node.node.op.attr["data_format"].s == b"NCHW"
            ):
                t_b_index = 1
            elif matched_node.node.op == "Conv3D" and matched_node.node.attr["data_format"].s == b"NDHWC":
                t_b_index = 4
            elif matched_node.node.op == "Conv3D" and matched_node.node.attr["data_format"].s == b"NCDHW":
                t_b_index = 1
            bias_add_length = op_b_node.attr["value"].tensor.tensor_shape.dim[t_b_index].size

            bias_add_content = [0.0] * bias_add_length

            bias_const_node = helper.create_constant_node(
                bias_const_node_name, bias_add_content, dtypes.float32, shape=[bias_add_length]
            )
            bias_node = helper.create_node("BiasAdd", bias_node_name, [target_node_name, bias_const_node_name])
            helper.set_attr_dtype(bias_node, "T", dtypes.float32)

            g.add_node(bias_node, target_node_name, next_node_names)
            g.add_node(bias_const_node, None, [bias_node_name])
            self.input_graph = g.dump_graph()
            self._parse_graph(self.input_graph)
            new_match_node_name = match_node_name[:2] + [bias_node_name] + match_node_name[2:]
            new_match_node_name = match_node_name[:2] + [bias_node_name] + match_node_name[2:]

        return new_match_node_name

    def apply_conv3d_add_addn_relu_fusion(self, match_node_name):
        """Apply Conv3D Add Addn Relu fusion.

        Dequantize + Conv3D + AddV2+ AddV2 + Relu + QuantizeV2
        """
        skip_node_name = match_node_name[2:]
        matched_node = self.node_name_mapping[match_node_name[1]]

        second_node = self.node_name_mapping[match_node_name[2]].node
        need_insert_dummy_biasadd = 1
        add_a_node_name = helper.node_name_from_input(second_node.input[0])
        add_a_node = self.node_name_mapping[add_a_node_name].node
        add_b_node_name = helper.node_name_from_input(second_node.input[1])
        add_b_node = self.node_name_mapping[add_b_node_name].node

        if add_a_node.op != "Const" and add_b_node.op == "Const":
            need_insert_dummy_biasadd = 0
        if need_insert_dummy_biasadd:
            new_match_node_name = self._insert_dummy_biasadd(match_node_name, matched_node)
            # after insert dummy biasadd, that is Conv3D+dummybiasadd+add*+add*+relu*
            return self.apply_conv3d_add_addn_fusion(new_match_node_name[:4] + [new_match_node_name[-1]])

        control_inputs, normal_inputs = self._get_node_input(matched_node.node.name)
        _, q_inputs = self._get_node_input(normal_inputs[0])
        _, q_weights_inputs = self._get_node_input(normal_inputs[1])
        quantizev2_weights_name = q_weights_inputs[0]

        _, weights_name = self._get_node_input(quantizev2_weights_name)
        weights_min_name = weights_name[1]
        weights_max_name = weights_name[2]

        third_node = self.node_name_mapping[match_node_name[3]].node
        sumadd_a_node_name = helper.node_name_from_input(third_node.input[0])
        sumadd_a_node = self.node_name_mapping[sumadd_a_node_name].node
        sumadd_b_node_name = helper.node_name_from_input(third_node.input[1])
        sumadd_b_node = self.node_name_mapping[sumadd_b_node_name].node

        if sumadd_a_node.op != "Const" and sumadd_b_node.op == "Const":
            return self.apply_conv3d_add_fusion(match_node_name[:3] + [match_node_name[-1]])

        forth_node = self.node_name_mapping[match_node_name[4]].node
        if third_node.op != "LeakyRelu" and not self._find_relu_node(matched_node.node):
            return self.apply_conv3d_add_fusion(match_node_name[:3] + [match_node_name[-1]])

        is_leakyrelu_add_fusion = third_node.op == "LeakyRelu" and forth_node.op.find("Add") != -1
        is_relu_add_fusion = third_node.op == "Relu" and forth_node.op.find("Add") != -1

        relu_offset = 0
        if is_leakyrelu_add_fusion or is_relu_add_fusion:
            relu_offset = 1
        if is_leakyrelu_add_fusion or is_relu_add_fusion:
            relu_node_name = match_node_name[3]
        else:
            relu_node_name = match_node_name[4]
        sum_index = (
            1
            if match_node_name[2 + relu_offset]
            == self.node_name_mapping[match_node_name[3 + relu_offset]].node.input[0]
            else 0
        )

        sum_node_name = self.node_name_mapping[match_node_name[3 + relu_offset]].node.input[sum_index]
        deq_node = self.node_name_mapping[sum_node_name].node
        if deq_node.op != "Dequantize" or deq_node.op.find("Quantize") != -1:
            return self.apply_conv3d_add_fusion(match_node_name[:3] + [match_node_name[-1]])

        add_node = self.node_name_mapping[match_node_name[2]].node
        original_add_input = self.node_name_mapping[add_node.input[1]].node
        if original_add_input.op == "Const":
            shape = tensor_util.MakeNdarray(original_add_input.attr["value"].tensor)
            if shape.ndim > 1 and shape.shape[:-1] == (1, 1, 1, 1):
                squeezed_value = np.squeeze(shape)
                squeezed_node = helper.create_constant_node(
                    match_node_name[1] + "_squeezed", squeezed_value, dtypes.float32
                )
                skip_node_name.append(add_node.input[1])
                add_node.input[1] = squeezed_node.name
                self.add_output_graph_node(squeezed_node)

        q_weights_name, q_weights_min_name, q_weights_max_name = self._intel_cpu_quantize_weight_eightbit(
            matched_node.node.op, self.node_name_mapping[weights_name[0]].node, self.per_channel
        )

        all_input_names = q_inputs[:1] + [q_weights_name] + q_inputs[1:]
        all_input_names.append(q_weights_min_name)
        all_input_names.append(q_weights_max_name)
        skip_node_name.append(normal_inputs[0])
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
                quantized_node_name = node.name + "_eightbit_quantized_conv"
                bias_node_name = self.node_name_mapping[match_node_name[2]].node.input[1]

                is_relu6 = self.node_name_mapping[relu_node_name].node.op == "Relu6"
                quantized_node_input_names = (
                    all_input_names[:2] + [bias_node_name] + all_input_names[2:] + [sum_node_name] + control_inputs
                )

                if sum_node_name.find("mul") != -1:
                    quantized_node_input_names = (
                        all_input_names[:2]
                        + [bias_node_name]
                        + [self.node_name_mapping[match_node_name[3 + relu_offset]].node.input[sum_index]]
                        + all_input_names[2:]
                        + control_inputs
                    )

                node_op = "_FusedQuantizedConv3D"

                quantized_conv_node = helper.create_node(node_op, quantized_node_name, quantized_node_input_names)
                helper.copy_attr(quantized_conv_node, "strides", node.attr["strides"])
                helper.copy_attr(quantized_conv_node, "padding", node.attr["padding"])
                helper.copy_attr(quantized_conv_node, "data_format", node.attr["data_format"])
                if "explicit_paddings" in node.attr:
                    helper.copy_attr(quantized_conv_node, "explicit_paddings", node.attr["explicit_paddings"])
                helper.copy_attr(quantized_conv_node, "dilations", node.attr["dilations"])
                input_data_type = dtypes.quint8 if self._find_relu_node(node) else dtypes.qint8
                helper.set_attr_dtype(quantized_conv_node, "Tinput", input_data_type)
                helper.set_attr_dtype(quantized_conv_node, "Tfilter", dtypes.qint8)
                helper.set_attr_dtype(quantized_conv_node, "out_type", dtypes.qint32)
                if "alpha" in self.node_name_mapping[relu_node_name].node.attr:
                    helper.copy_attr(
                        quantized_conv_node, "alpha", self.node_name_mapping[relu_node_name].node.attr["alpha"]
                    )
                helper.set_attr_string_list(quantized_conv_node, "fused_ops", [b"BiasAdd", b"Sum", b"Relu"])
                helper.set_attr_dtype(quantized_conv_node, "Tbias", dtypes.float32)
                # if self.device == 'gpu' else dtypes.qint32)
                helper.set_attr_dtype(quantized_conv_node, "Tsummand", dtypes.qint32)
                if is_leakyrelu_add_fusion:
                    helper.set_attr_string_list(quantized_conv_node, "fused_ops", [b"BiasAdd", b"LeakyRelu", b"Sum"])
                elif is_relu_add_fusion:
                    helper.set_attr_string_list(quantized_conv_node, "fused_ops", [b"BiasAdd", b"Relu", b"Sum"])

                helper.set_attr_type_list(
                    quantized_conv_node,
                    "Thost_inputs",
                    [
                        input_data_type.as_datatype_enum,
                        dtypes.qint8.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,  # both cpu and gpu use float32 in New API
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ],
                )
                helper.set_attr_type_list(
                    quantized_conv_node,
                    "Thost_outputs",
                    [
                        dtypes.qint32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ],
                )

                self.add_output_graph_node(quantized_conv_node)

                if is_leakyrelu_add_fusion:
                    quantize_down_name = self._add_quantize_down_nodes(node, quantized_node_name, dtypes.qint8, False)
                    self._intel_cpu_add_dequantize_result_node(
                        quantize_down_name,
                        match_node_name[4],
                        dtype=dtypes.qint8,
                        performance_only=self.performance_only,
                    )
                else:
                    dtype = dtypes.quint8
                    if [
                        i
                        for i in self.node_name_mapping[relu_node_name].output
                        if "FusedBatchNorm" in self.node_name_mapping[i].node.op and i in self.op_wise_config_name_list
                    ]:
                        dtype = dtypes.qint8
                    quantize_down_name = self._add_quantize_down_nodes(node, quantized_node_name, dtype, is_relu6)
                    self._intel_cpu_add_dequantize_result_node(
                        quantize_down_name, relu_node_name, dtype, performance_only=self.performance_only
                    )

            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)

    def apply_conv3d_add_addn_fusion(self, match_node_name):
        """Apply Conv3D Add Addn fusion.

        Dequantize + Conv3D + BiasAdd + Add + QuantizeV2
        Dequantize + Conv3D + BiasAdd + AddV2 + QuantizeV2
        Dequantize + Conv3D + AddV2 + AddV2 + QuantizeV2
        """
        skip_node_name = match_node_name[2:]
        matched_node = self.node_name_mapping[match_node_name[1]]

        second_node = self.node_name_mapping[match_node_name[2]].node
        need_insert_dummy_biasadd = 1
        add_a_node_name = helper.node_name_from_input(second_node.input[0])
        add_a_node = self.node_name_mapping[add_a_node_name].node
        add_b_node_name = helper.node_name_from_input(second_node.input[1])
        add_b_node = self.node_name_mapping[add_b_node_name].node

        if add_a_node.op != "Const" and add_b_node.op == "Const":
            need_insert_dummy_biasadd = 0
        if need_insert_dummy_biasadd:
            new_match_node_name = self._insert_dummy_biasadd(match_node_name, matched_node)
            # after insert dummy biasadd, that is Conv+dummybiasadd+add*+add*
            return self.apply_conv3d_add_addn_fusion(new_match_node_name[:4] + [new_match_node_name[-1]])

        control_inputs, normal_inputs = self._get_node_input(matched_node.node.name)
        _, q_inputs = self._get_node_input(normal_inputs[0])
        _, q_weights_inputs = self._get_node_input(normal_inputs[1])
        quantizev2_weights_name = q_weights_inputs[0]

        _, weights_name = self._get_node_input(quantizev2_weights_name)
        weights_min_name = weights_name[1]
        weights_max_name = weights_name[2]

        third_node = self.node_name_mapping[match_node_name[3]].node
        sumadd_a_node_name = helper.node_name_from_input(third_node.input[0])
        sumadd_a_node = self.node_name_mapping[sumadd_a_node_name].node
        sumadd_b_node_name = helper.node_name_from_input(third_node.input[1])
        sumadd_b_node = self.node_name_mapping[sumadd_b_node_name].node
        if sumadd_a_node.op != "Const" and sumadd_b_node.op == "Const":
            return self.apply_conv3d_add_fusion(match_node_name[:3] + [match_node_name[-1]])

        sum_index = 1 if match_node_name[2] == self.node_name_mapping[match_node_name[3]].node.input[0] else 0
        sum_node_name = self.node_name_mapping[match_node_name[3]].node.input[sum_index]
        deq_node = self.node_name_mapping[sum_node_name].node
        if deq_node.op != "Dequantize" or deq_node.op.find("Quantize") != -1:
            return self.apply_conv3d_add_fusion(match_node_name[:3] + [match_node_name[-1]])

        add_node = self.node_name_mapping[match_node_name[2]].node
        original_add_input = self.node_name_mapping[add_node.input[1]].node
        if original_add_input.op == "Const":
            shape = tensor_util.MakeNdarray(original_add_input.attr["value"].tensor)
            if shape.ndim > 1 and shape.shape[:-1] == (1, 1, 1, 1):
                squeezed_value = np.squeeze(shape)
                squeezed_node = helper.create_constant_node(
                    match_node_name[2] + "_squeezed", squeezed_value, dtypes.float32
                )
                skip_node_name.append(add_node.input[1])
                add_node.input[1] = squeezed_node.name
                self.add_output_graph_node(squeezed_node)

        q_weights_name, q_weights_min_name, q_weights_max_name = self._intel_cpu_quantize_weight_eightbit(
            matched_node.node.op, self.node_name_mapping[weights_name[0]].node, self.per_channel
        )

        all_input_names = q_inputs[:1] + [q_weights_name] + q_inputs[1:]
        all_input_names.append(q_weights_min_name)
        all_input_names.append(q_weights_max_name)
        skip_node_name.append(normal_inputs[0])
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
                quantized_node_name = node.name + "_eightbit_quantized_conv"
                bias_node_name = self.node_name_mapping[match_node_name[2]].node.input[1]

                quantized_node_input_names = (
                    all_input_names[:2] + [bias_node_name] + all_input_names[2:] + [sum_node_name] + control_inputs
                )
                node_op = "_FusedQuantizedConv3D"

                quantized_conv_node = helper.create_node(node_op, quantized_node_name, quantized_node_input_names)
                helper.copy_attr(quantized_conv_node, "strides", node.attr["strides"])
                helper.copy_attr(quantized_conv_node, "padding", node.attr["padding"])
                helper.copy_attr(quantized_conv_node, "data_format", node.attr["data_format"])
                if "explicit_paddings" in node.attr:
                    helper.copy_attr(quantized_conv_node, "explicit_paddings", node.attr["explicit_paddings"])
                helper.copy_attr(quantized_conv_node, "dilations", node.attr["dilations"])
                input_data_type = dtypes.quint8 if self._find_relu_node(node) else dtypes.qint8
                helper.set_attr_dtype(quantized_conv_node, "Tinput", input_data_type)
                helper.set_attr_dtype(quantized_conv_node, "Tfilter", dtypes.qint8)
                helper.set_attr_dtype(quantized_conv_node, "out_type", dtypes.qint32)
                helper.set_attr_string_list(quantized_conv_node, "fused_ops", [b"BiasAdd", b"Sum"])
                helper.set_attr_dtype(quantized_conv_node, "Tbias", dtypes.float32)
                # if self.device == 'gpu' else dtypes.qint32)
                helper.set_attr_dtype(quantized_conv_node, "Tsummand", dtypes.qint32)

                helper.set_attr_type_list(
                    quantized_conv_node,
                    "Thost_inputs",
                    [
                        input_data_type.as_datatype_enum,
                        dtypes.qint8.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,  # if self.device == 'gpu' else dtypes.qint32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ],
                )
                helper.set_attr_type_list(
                    quantized_conv_node,
                    "Thost_outputs",
                    [
                        dtypes.qint32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ],
                )

                self.add_output_graph_node(quantized_conv_node)
                quantize_down_name = self._add_quantize_down_nodes(node, quantized_node_name, dtypes.qint8, False)
                self._intel_cpu_add_dequantize_result_node(
                    quantize_down_name, match_node_name[3], dtype=dtypes.qint8, performance_only=self.performance_only
                )
            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)

    def apply_conv3d_add_relu_fusion(self, match_node_name):
        """Apply Conv3D Add Relu fusion.

        Dequantize + Conv3D + Add + Relu + QuantizeV2
        Dequantize + Conv3D + AddV2 + Relu + QuantizeV2
        """
        skip_node_name = match_node_name[2:]
        matched_node = self.node_name_mapping[match_node_name[1]]

        second_node = self.node_name_mapping[match_node_name[2]].node
        if second_node.op in ("Relu", "Relu6", "LeakyRelu", "Elu"):
            new_match_node_name = self._insert_dummy_biasadd(match_node_name, matched_node)
            return self.apply_conv3d_add_relu_fusion(new_match_node_name)

        need_insert_dummy_biasadd = 1
        add_a_node_name = helper.node_name_from_input(second_node.input[0])
        add_a_node = self.node_name_mapping[add_a_node_name].node
        add_b_node_name = helper.node_name_from_input(second_node.input[1])
        add_b_node = self.node_name_mapping[add_b_node_name].node

        if add_a_node.op != "Const" and add_b_node.op == "Const":
            need_insert_dummy_biasadd = 0
        if need_insert_dummy_biasadd:
            new_match_node_name = self._insert_dummy_biasadd(match_node_name, matched_node)
            # after insert dummy biasadd, that is Conv+dummybiasadd+add*+relu*
            return self.apply_conv3d_add_addn_relu_fusion(new_match_node_name)

        control_inputs, normal_inputs = self._get_node_input(matched_node.node.name)
        _, q_inputs = self._get_node_input(normal_inputs[0])
        _, q_weights_inputs = self._get_node_input(normal_inputs[1])
        quantizev2_weights_name = q_weights_inputs[0]

        _, weights_name = self._get_node_input(quantizev2_weights_name)
        weights_min_name = weights_name[1]
        weights_max_name = weights_name[2]

        # third_node = self.node_name_mapping[match_node_name[3]].node
        # if third_node.op != 'LeakyRelu' and not self._find_relu_node(matched_node.node):
        #     return self.apply_conv3d_add_fusion(match_node_name[:3] + [match_node_name[-1]])

        add_node = self.node_name_mapping[match_node_name[2]].node
        original_add_input = self.node_name_mapping[add_node.input[1]].node
        if original_add_input.op == "Const":
            shape = tensor_util.MakeNdarray(original_add_input.attr["value"].tensor)
            if shape.ndim > 1 and shape.shape[:-1] == (1, 1, 1, 1):
                squeezed_value = np.squeeze(shape)
                squeezed_node = helper.create_constant_node(
                    match_node_name[1] + "_squeezed", squeezed_value, dtypes.float32
                )
                skip_node_name.append(add_node.input[1])
                add_node.input[1] = squeezed_node.name
                self.add_output_graph_node(squeezed_node)

        q_weights_name, q_weights_min_name, q_weights_max_name = self._intel_cpu_quantize_weight_eightbit(
            matched_node.node.op, self.node_name_mapping[weights_name[0]].node, self.per_channel
        )

        all_input_names = q_inputs[:1] + [q_weights_name] + q_inputs[1:]
        all_input_names.append(q_weights_min_name)
        all_input_names.append(q_weights_max_name)
        skip_node_name.append(normal_inputs[0])
        skip_node_name.append(normal_inputs[1])
        skip_node_name.append(weights_name[0])
        skip_node_name.append(weights_min_name)
        skip_node_name.append(weights_max_name)
        skip_node_name.append(quantizev2_weights_name)

        for _, node in enumerate(self.input_graph.node):
            if node.name in skip_node_name:
                self.logger.debug("skip node {}".format(node.name))
            elif node.name == match_node_name[1]:
                # actually DepthwiseConv3dNative is not supported in intel-tf 2.8 yet
                postfix = "_eightbit_quantized_depthwise_conv3d"
                if node.op == "Conv3D":
                    postfix = "_eightbit_quantized_conv3d"
                quantized_node_name = node.name + postfix

                bias_node_name = self.node_name_mapping[match_node_name[2]].node.input[1]
                relu_node_name = match_node_name[3]
                is_relu6 = self.node_name_mapping[relu_node_name].node.op == "Relu6"
                quantized_node_input_names = (
                    all_input_names[:2] + [bias_node_name] + all_input_names[2:] + control_inputs
                )

                is_leakyrelu = self.node_name_mapping[relu_node_name].node.op == "LeakyRelu"

                quantized_conv_node = helper.create_node(
                    "_FusedQuantizedConv3D", quantized_node_name, quantized_node_input_names
                )

                helper.copy_attr(quantized_conv_node, "strides", node.attr["strides"])
                helper.copy_attr(quantized_conv_node, "padding", node.attr["padding"])
                helper.copy_attr(quantized_conv_node, "data_format", node.attr["data_format"])
                if "alpha" in self.node_name_mapping[relu_node_name].node.attr:
                    helper.copy_attr(
                        quantized_conv_node, "alpha", self.node_name_mapping[relu_node_name].node.attr["alpha"]
                    )
                if node.op != "DepthwiseConv3dNative" and "explicit_paddings" in node.attr:
                    helper.copy_attr(quantized_conv_node, "explicit_paddings", node.attr["explicit_paddings"])
                helper.copy_attr(quantized_conv_node, "dilations", node.attr["dilations"])
                input_data_type = dtypes.quint8 if self._find_relu_node(node) else dtypes.qint8
                helper.set_attr_dtype(quantized_conv_node, "Tinput", input_data_type)
                helper.set_attr_dtype(quantized_conv_node, "Tfilter", dtypes.qint8)
                helper.set_attr_dtype(quantized_conv_node, "Tsummand", dtypes.qint32)
                # helper.set_attr_string(quantized_conv_node, '_kernel', b'QuantizedMklOp')
                helper.set_attr_dtype(quantized_conv_node, "out_type", dtypes.qint32)
                # helper.set_attr_dtype(quantized_conv_node, "alpha", dtypes.quint8)
                helper.set_attr_dtype(quantized_conv_node, "Tbias", dtypes.float32)
                # if self.device == 'gpu' else dtypes.qint32)
                if self.node_name_mapping[relu_node_name].node.op == "LeakyRelu":
                    helper.set_attr_string_list(quantized_conv_node, "fused_ops", [b"BiasAdd", b"LeakyRelu"])
                elif self.node_name_mapping[relu_node_name].node.op == "Elu":
                    helper.set_attr_string_list(quantized_conv_node, "fused_ops", [b"BiasAdd", b"Elu"])
                else:
                    helper.set_attr_string_list(quantized_conv_node, "fused_ops", [b"BiasAdd", b"Relu"])
                helper.set_attr_type_list(
                    quantized_conv_node,
                    "Thost_inputs",
                    [
                        input_data_type.as_datatype_enum,
                        dtypes.qint8.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,  # if self.device == 'gpu' else dtypes.qint32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ],
                )
                helper.set_attr_type_list(
                    quantized_conv_node,
                    "Thost_outputs",
                    [
                        dtypes.qint32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ],
                )

                self.add_output_graph_node(quantized_conv_node)
                if not is_leakyrelu:
                    dtype = dtypes.quint8
                    if [
                        i
                        for i in self.node_name_mapping[relu_node_name].output
                        if "FusedBatchNorm" in self.node_name_mapping[i].node.op and i in self.op_wise_config_name_list
                    ]:
                        dtype = dtypes.qint8
                    quantize_down_name = self._add_quantize_down_nodes(node, quantized_node_name, dtype, is_relu6)
                    self._intel_cpu_add_dequantize_result_node(
                        quantize_down_name, relu_node_name, dtype, performance_only=self.performance_only
                    )
                else:
                    quantize_down_name = self._add_quantize_down_nodes(node, quantized_node_name, dtypes.qint8, False)
                    self._intel_cpu_add_dequantize_result_node(
                        quantize_down_name, relu_node_name, dtype=dtypes.qint8, performance_only=self.performance_only
                    )

            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)

    def apply_conv3d_add_fusion(self, match_node_name):
        """Apply Conv3D Add fusion.

        Dequantize + Conv3D + BiasAdd + QuantizeV2
        Dequantize + Conv3D + Add + QuantizeV2
        Dequantize + Conv3D + AddV2 + QuantizeV2
        """
        skip_node_name = match_node_name[2:]
        matched_node = self.node_name_mapping[match_node_name[1]]

        second_node = self.node_name_mapping[match_node_name[2]].node
        need_insert_dummy_biasadd = 1
        add_a_node_name = helper.node_name_from_input(second_node.input[0])
        add_a_node = self.node_name_mapping[add_a_node_name].node
        add_b_node_name = helper.node_name_from_input(second_node.input[1])
        add_b_node = self.node_name_mapping[add_b_node_name].node
        if add_a_node.op != "Const" and add_b_node.op == "Const":
            need_insert_dummy_biasadd = 0
        if need_insert_dummy_biasadd:
            new_match_node_name = self._insert_dummy_biasadd(match_node_name, matched_node)
            return self.apply_conv3d_add_addn_fusion(new_match_node_name)

        _, normal_inputs = self._get_node_input(matched_node.node.name)
        _, q_inputs = self._get_node_input(normal_inputs[0])
        _, q_weights_inputs = self._get_node_input(normal_inputs[1])
        quantizev2_weights_name = q_weights_inputs[0]

        _, weights_name = self._get_node_input(quantizev2_weights_name)
        weights_min_name = weights_name[1]
        weights_max_name = weights_name[2]

        add_node = self.node_name_mapping[match_node_name[2]].node
        original_add_input = self.node_name_mapping[add_node.input[1]].node
        if original_add_input.op == "Const":
            shape = tensor_util.MakeNdarray(original_add_input.attr["value"].tensor)
            if shape.ndim > 1 and shape.shape[:-1] == (1, 1, 1, 1):
                squeezed_value = np.squeeze(shape)
                squeezed_node = helper.create_constant_node(
                    match_node_name[2] + "_squeezed", squeezed_value, dtypes.float32
                )
                skip_node_name.append(add_node.input[1])
                add_node.input[1] = squeezed_node.name
                self.add_output_graph_node(squeezed_node)

        q_weights_name, q_weights_min_name, q_weights_max_name = self._intel_cpu_quantize_weight_eightbit(
            matched_node.node.op, self.node_name_mapping[weights_name[0]].node, self.per_channel
        )

        all_input_names = q_inputs[:1] + [q_weights_name] + q_inputs[1:]
        all_input_names.append(q_weights_min_name)
        all_input_names.append(q_weights_max_name)
        skip_node_name.append(normal_inputs[0])
        skip_node_name.append(normal_inputs[1])
        skip_node_name.append(weights_name[0])
        skip_node_name.append(weights_min_name)
        skip_node_name.append(weights_max_name)
        skip_node_name.append(quantizev2_weights_name)

        for _, node in enumerate(self.input_graph.node):
            if node.name in skip_node_name:
                self.logger.debug("skip node {}".format(node.name))
            elif node.name == match_node_name[1]:
                # actually DepthwiseConv3dNative is not supported in intel-tf 2.8 yet
                postfix = "_eightbit_quantized_depthwise_conv3d"
                if node.op == "Conv3D":
                    postfix = "_eightbit_quantized_conv3d"
                quantized_node_name = node.name + postfix

                bias_node_name = self.node_name_mapping[match_node_name[2]].node.input[1]
                quantized_node_input_names = all_input_names[:2] + [bias_node_name] + all_input_names[2:]
                if node.op == "Conv3D":
                    quantized_conv_node = helper.create_node(
                        "_FusedQuantizedConv3D", quantized_node_name, quantized_node_input_names
                    )

                helper.copy_attr(quantized_conv_node, "strides", node.attr["strides"])
                helper.copy_attr(quantized_conv_node, "padding", node.attr["padding"])
                helper.copy_attr(quantized_conv_node, "data_format", node.attr["data_format"])
                if node.op != "DepthwiseConv3dNative" and "explicit_paddings" in node.attr:
                    helper.copy_attr(quantized_conv_node, "explicit_paddings", node.attr["explicit_paddings"])
                helper.copy_attr(quantized_conv_node, "dilations", node.attr["dilations"])
                input_data_type = dtypes.quint8 if self._find_relu_node(node) else dtypes.qint8
                helper.set_attr_dtype(quantized_conv_node, "Tinput", input_data_type)
                helper.set_attr_dtype(quantized_conv_node, "Tfilter", dtypes.qint8)
                helper.set_attr_dtype(quantized_conv_node, "Tbias", dtypes.float32)
                # if self.device == 'gpu' else dtypes.qint32)
                helper.set_attr_dtype(quantized_conv_node, "Tsummand", dtypes.qint32)
                # helper.set_attr_string(quantized_conv_node, '_kernel', b'QuantizedMklOp')
                helper.set_attr_dtype(quantized_conv_node, "out_type", dtypes.qint32)
                helper.set_attr_string_list(quantized_conv_node, "fused_ops", [b"BiasAdd"])

                helper.set_attr_type_list(
                    quantized_conv_node,
                    "Thost_inputs",
                    [
                        input_data_type.as_datatype_enum,
                        dtypes.qint8.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,  # if self.device == 'gpu' else dtypes.qint32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ],
                )
                helper.set_attr_type_list(
                    quantized_conv_node,
                    "Thost_outputs",
                    [
                        dtypes.qint32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ],
                )

                self.add_output_graph_node(quantized_conv_node)

                quantize_down_name = self._add_quantize_down_nodes(node, quantized_node_name, dtypes.qint8)

                self._intel_cpu_add_dequantize_result_node(
                    quantize_down_name, match_node_name[2], dtypes.qint8, performance_only=self.performance_only
                )
            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)

    def apply_conv3d_single_fusion(self, match_node_name):
        """Apply Conv3D single fusion.

        Dequantize + Conv3D + QuantizeV2
        """
        skip_node_name = match_node_name[2:]
        matched_node = self.node_name_mapping[match_node_name[1]]
        _, normal_inputs = self._get_node_input(matched_node.node.name)
        _, q_inputs = self._get_node_input(normal_inputs[0])
        _, q_weights_inputs = self._get_node_input(normal_inputs[1])
        quantizev2_weights_name = q_weights_inputs[0]

        _, weights_name = self._get_node_input(quantizev2_weights_name)
        weights_min_name = weights_name[1]
        weights_max_name = weights_name[2]

        q_weights_name, q_weights_min_name, q_weights_max_name = self._intel_cpu_quantize_weight_eightbit(
            matched_node.node.op, self.node_name_mapping[weights_name[0]].node, self.per_channel
        )

        all_input_names = q_inputs[:1] + [q_weights_name] + q_inputs[1:]
        all_input_names.append(q_weights_min_name)
        all_input_names.append(q_weights_max_name)
        skip_node_name.append(normal_inputs[0])
        skip_node_name.append(normal_inputs[1])
        skip_node_name.append(weights_name[0])
        skip_node_name.append(weights_min_name)
        skip_node_name.append(weights_max_name)
        skip_node_name.append(quantizev2_weights_name)

        for _, node in enumerate(self.input_graph.node):
            if node.name in skip_node_name:
                self.logger.debug("skip node {}".format(node.name))
            elif node.name == match_node_name[1]:
                # actually DepthwiseConv3dNative is not supported in intel-tf 2.8 yet
                postfix = "_eightbit_quantized_depthwise_conv3d"
                if node.op == "Conv3D":
                    postfix = "_eightbit_quantized_conv3d"
                quantized_node_name = node.name + postfix
                quantized_conv_node = helper.create_node(
                    "_FusedQuantizedConv3D",
                    quantized_node_name,
                    all_input_names,
                )

                helper.copy_attr(quantized_conv_node, "strides", node.attr["strides"])
                helper.copy_attr(quantized_conv_node, "padding", node.attr["padding"])
                helper.copy_attr(quantized_conv_node, "data_format", node.attr["data_format"])
                if node.op != "DepthwiseConv3dNative" and "explicit_paddings" in node.attr:
                    helper.copy_attr(quantized_conv_node, "explicit_paddings", node.attr["explicit_paddings"])
                helper.copy_attr(quantized_conv_node, "dilations", node.attr["dilations"])
                input_data_type = dtypes.quint8 if self._find_relu_node(node) else dtypes.qint8
                helper.set_attr_dtype(quantized_conv_node, "Tinput", input_data_type)
                helper.set_attr_dtype(quantized_conv_node, "Tfilter", dtypes.qint8)
                helper.set_attr_dtype(quantized_conv_node, "out_type", dtypes.qint32)
                helper.set_attr_dtype(quantized_conv_node, "Tsummand", dtypes.qint32)
                helper.set_attr_dtype(quantized_conv_node, "Tbias", dtypes.float32)
                # if self.device == 'gpu' else dtypes.qint32)
                # helper.set_attr_string(quantized_conv_node, '_kernel', b'QuantizedMklOp')
                helper.set_attr_string_list(quantized_conv_node, "fused_ops", [])

                helper.set_attr_type_list(
                    quantized_conv_node,
                    "Thost_inputs",
                    [
                        input_data_type.as_datatype_enum,
                        dtypes.qint8.as_datatype_enum,
                        # dtypes.float32.as_datatype_enum if self.device == 'gpu' else dtypes.qint32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ],
                )
                helper.set_attr_type_list(
                    quantized_conv_node,
                    "Thost_outputs",
                    [
                        dtypes.qint32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ],
                )
                self.add_output_graph_node(quantized_conv_node)
                quantize_down_name = self._add_quantize_down_nodes(node, quantized_node_name, dtypes.qint8)
                self._intel_cpu_add_dequantize_result_node(
                    quantize_down_name, node.name, dtypes.qint8, performance_only=self.performance_only
                )
            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)

    def apply_newly_conv_biasadd_relu_fusion(self, match_node_name):
        """Apply Conv2D BiasAdd Relu fusion.

        Dequantize + Conv2D + AddN + Relu + QuantizeV2
        Dequantize + Conv2D + AddN + Relu6 + QuantizeV2
        Dequantize + Conv2D + AddV2 + Relu + QuantizeV2
        Dequantize + Conv2D + AddV2 + Relu6 + QuantizeV2
        Dequantize + Conv2D + BiasAdd + Relu6 + QuantizeV2
        Dequantize + Conv2D + Relu6 + QuantizeV2
        Dequantize + Conv2D + BiasAdd + Relu + QuantizeV2
        Dequantize + Conv2D + Relu + QuantizeV2
        Dequantize + Conv2D + BiasAdd + Elu + QuantizeV2
        Dequantize + Conv2D + Elu + QuantizeV2
        Dequantize + Conv2D + BiasAdd + LeakyRelu + QuantizeV2
        Dequantize + Conv2D + LeakyRelu + QuantizeV2
        Dequantize + Conv2D + BiasAdd + Sigmoid + QuantizeV2
        Dequantize + Conv2D + Sigmoid + QuantizeV2
        Dequantize + Conv2D + Add + Relu6 + QuantizeV2
        Dequantize + Conv2D + Add + Relu + QuantizeV2
        Dequantize + DepthwiseConv2dNative + Add + Relu6 + QuantizeV2
        Dequantize + DepthwiseConv2dNative + BiasAdd + Relu + QuantizeV2
        Dequantize + DepthwiseConv2dNative + Relu + QuantizeV2
        Dequantize + DepthwiseConv2dNative + Relu6 + QuantizeV2
        Dequantize + DepthwiseConv2dNative + BiasAdd + LeakyRelu + QuantizeV2
        Dequantize + DepthwiseConv2dNative + LeakyRelu + QuantizeV2
        Dequantize + DepthwiseConv2dNative + BiasAdd + Relu6 + QuantizeV2
        """
        skip_node_name = match_node_name[2:]
        matched_node = self.node_name_mapping[match_node_name[1]]

        second_node = self.node_name_mapping[match_node_name[2]].node
        if second_node.op in ("Relu", "Relu6", "LeakyRelu", "Elu", "Sigmoid"):
            new_match_node_name = self._insert_dummy_biasadd(match_node_name, matched_node)
            return self.apply_newly_conv_biasadd_relu_fusion(new_match_node_name)

        need_insert_dummy_biasadd = 1
        add_a_node_name = helper.node_name_from_input(second_node.input[0])
        add_a_node = self.node_name_mapping[add_a_node_name].node
        add_b_node_name = helper.node_name_from_input(second_node.input[1])
        add_b_node = self.node_name_mapping[add_b_node_name].node

        if add_a_node.op != "Const" and add_b_node.op == "Const":
            need_insert_dummy_biasadd = 0
        if need_insert_dummy_biasadd:
            new_match_node_name = self._insert_dummy_biasadd(match_node_name, matched_node)
            # after insert dummy biasadd, that is Conv+dummybiasadd+add*+relu*
            return self.apply_newly_conv_biasadd_addn_relu_fusion(new_match_node_name)

        control_inputs, normal_inputs = self._get_node_input(matched_node.node.name)
        _, q_inputs = self._get_node_input(normal_inputs[0])
        _, q_weights_inputs = self._get_node_input(normal_inputs[1])

        quantizev2_weights_name = q_weights_inputs[0]
        _, weights_name = self._get_node_input(quantizev2_weights_name)
        weights_min_name = weights_name[1]
        weights_max_name = weights_name[2]

        q_weights_name, q_weights_min_name, q_weights_max_name = self._intel_cpu_quantize_weight_eightbit(
            matched_node.node.op, self.node_name_mapping[weights_name[0]].node, self.per_channel
        )

        all_input_names = q_inputs[:1] + [q_weights_name] + q_inputs[1:]
        all_input_names.append(q_weights_min_name)
        all_input_names.append(q_weights_max_name)
        skip_node_name.append(normal_inputs[0])
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
                quantized_node_name = node.name + "_eightbit_quantized_depthwise_conv"
                if node.op == "Conv2D":
                    quantized_node_name = node.name + "_eightbit_quantized_conv"

                bias_node_name = self.node_name_mapping[match_node_name[2]].node.input[1]
                relu_node_name = match_node_name[3]
                is_relu6 = self.node_name_mapping[relu_node_name].node.op == "Relu6"
                quantized_node_input_names = (
                    all_input_names[:2] + [bias_node_name] + all_input_names[2:] + control_inputs
                )
                is_leakyrelu = self.node_name_mapping[relu_node_name].node.op == "LeakyRelu"
                is_elu = self.node_name_mapping[relu_node_name].node.op == "Elu"
                is_sigmoid = self.node_name_mapping[relu_node_name].node.op == "Sigmoid"

                node_op = "_FusedQuantizedDepthwiseConv2D"
                if node.op == "Conv2D":
                    node_op = "_FusedQuantizedConv2D"
                quantized_conv_node = helper.create_node(node_op, quantized_node_name, quantized_node_input_names)

                helper.copy_attr(quantized_conv_node, "strides", node.attr["strides"])
                helper.copy_attr(quantized_conv_node, "padding", node.attr["padding"])
                helper.copy_attr(quantized_conv_node, "data_format", node.attr["data_format"])
                if "alpha" in self.node_name_mapping[relu_node_name].node.attr:
                    helper.copy_attr(
                        quantized_conv_node, "alpha", self.node_name_mapping[relu_node_name].node.attr["alpha"]
                    )
                if "explicit_paddings" in node.attr:
                    helper.copy_attr(quantized_conv_node, "explicit_paddings", node.attr["explicit_paddings"])
                helper.copy_attr(quantized_conv_node, "dilations", node.attr["dilations"])
                input_data_type = dtypes.quint8 if self._find_relu_node(node) else dtypes.qint8
                helper.set_attr_dtype(quantized_conv_node, "Tinput", input_data_type)
                helper.set_attr_dtype(quantized_conv_node, "Tfilter", dtypes.qint8)
                helper.set_attr_dtype(quantized_conv_node, "Tsummand", dtypes.qint32)
                # helper.set_attr_string(quantized_conv_node, '_kernel', b'QuantizedMklOp')
                helper.set_attr_dtype(quantized_conv_node, "out_type", dtypes.qint32)
                # helper.set_attr_dtype(quantized_conv_node, "alpha", dtypes.quint8)
                helper.set_attr_dtype(quantized_conv_node, "Tbias", dtypes.float32)
                # if self.device == 'gpu' else dtypes.qint32)
                fused_ops = [b"BiasAdd", b"Relu"]
                if is_leakyrelu:
                    fused_ops = [b"BiasAdd", b"LeakyRelu"]
                if is_elu:
                    fused_ops = [b"BiasAdd", b"Elu"]
                if is_sigmoid:
                    fused_ops = [b"BiasAdd", b"Sigmoid"]
                helper.set_attr_string_list(quantized_conv_node, "fused_ops", fused_ops)
                helper.set_attr_type_list(
                    quantized_conv_node,
                    "Thost_inputs",
                    [
                        input_data_type.as_datatype_enum,
                        dtypes.qint8.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,  # if self.device == 'gpu' else dtypes.qint32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ],
                )
                helper.set_attr_type_list(
                    quantized_conv_node,
                    "Thost_outputs",
                    [
                        dtypes.qint32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ],
                )
                self.add_output_graph_node(quantized_conv_node)

                if not is_leakyrelu:
                    dtype = dtypes.quint8
                    if [
                        i
                        for i in self.node_name_mapping[relu_node_name].output
                        if "FusedBatchNorm" in self.node_name_mapping[i].node.op and i in self.op_wise_config_name_list
                    ]:
                        dtype = dtypes.qint8
                    quantize_down_name = self._add_quantize_down_nodes(node, quantized_node_name, dtype, is_relu6)
                    self._intel_cpu_add_dequantize_result_node(
                        quantize_down_name, relu_node_name, dtype, performance_only=self.performance_only
                    )
                else:
                    quantize_down_name = self._add_quantize_down_nodes(node, quantized_node_name, dtypes.qint8, False)
                    self._intel_cpu_add_dequantize_result_node(
                        quantize_down_name, relu_node_name, dtype=dtypes.qint8, performance_only=self.performance_only
                    )
            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)

    def apply_newly_conv_biasadd_fusion(self, match_node_name):
        """Apply Conv2D BiasAdd fusion.

        Dequantize + Conv2D + Biasadd + QuantizeV2
        Dequantize + DepthwiseConv2dNative + BiasAdd + QuantizeV2
        Dequantize + Conv2D + Add + QuantizeV2
        Dequantize + Conv2D + AddV2 + QuantizeV2
        """
        skip_node_name = match_node_name[2:]
        matched_node = self.node_name_mapping[match_node_name[1]]

        second_node = self.node_name_mapping[match_node_name[2]].node
        need_insert_dummy_biasadd = 1
        add_a_node_name = helper.node_name_from_input(second_node.input[0])
        add_a_node = self.node_name_mapping[add_a_node_name].node
        add_b_node_name = helper.node_name_from_input(second_node.input[1])
        add_b_node = self.node_name_mapping[add_b_node_name].node

        if add_a_node.op != "Const" and add_b_node.op == "Const":
            need_insert_dummy_biasadd = 0
        if need_insert_dummy_biasadd:
            new_match_node_name = self._insert_dummy_biasadd(match_node_name, matched_node)
            # after insert dummy biasadd, that is Conv+dummybiasadd+add
            return self.apply_newly_conv_biasadd_addn_fusion(new_match_node_name)

        control_inputs, normal_inputs = self._get_node_input(matched_node.node.name)
        _, q_inputs = self._get_node_input(normal_inputs[0])
        _, q_weights_inputs = self._get_node_input(normal_inputs[1])

        quantizev2_weights_name = q_weights_inputs[0]
        _, weights_name = self._get_node_input(quantizev2_weights_name)
        weights_min_name = weights_name[1]
        weights_max_name = weights_name[2]

        q_weights_name, q_weights_min_name, q_weights_max_name = self._intel_cpu_quantize_weight_eightbit(
            matched_node.node.op, self.node_name_mapping[weights_name[0]].node, self.per_channel
        )

        all_input_names = q_inputs[:1] + [q_weights_name] + q_inputs[1:]
        all_input_names.append(q_weights_min_name)
        all_input_names.append(q_weights_max_name)
        skip_node_name.append(normal_inputs[0])
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
                quantized_node_name = node.name + "_eightbit_quantized_depthwise_conv"
                if node.op == "Conv2D":
                    quantized_node_name = node.name + "_eightbit_quantized_conv"

                bias_node_name = self.node_name_mapping[match_node_name[2]].node.input[1]
                quantized_node_input_names = (
                    all_input_names[:2] + [bias_node_name] + all_input_names[2:] + control_inputs
                )

                node_op = "_FusedQuantizedConv2D" if node.op == "Conv2D" else "_FusedQuantizedDepthwiseConv2D"
                quantized_conv_node = helper.create_node(node_op, quantized_node_name, quantized_node_input_names)

                helper.copy_attr(quantized_conv_node, "strides", node.attr["strides"])
                helper.copy_attr(quantized_conv_node, "padding", node.attr["padding"])
                helper.copy_attr(quantized_conv_node, "data_format", node.attr["data_format"])
                if "explicit_paddings" in node.attr:
                    helper.copy_attr(quantized_conv_node, "explicit_paddings", node.attr["explicit_paddings"])
                helper.copy_attr(quantized_conv_node, "dilations", node.attr["dilations"])
                input_data_type = dtypes.quint8 if self._find_relu_node(node) else dtypes.qint8
                helper.set_attr_dtype(quantized_conv_node, "Tinput", input_data_type)
                helper.set_attr_dtype(quantized_conv_node, "Tfilter", dtypes.qint8)
                helper.set_attr_dtype(quantized_conv_node, "Tsummand", dtypes.qint32)
                # helper.set_attr_string(quantized_conv_node, '_kernel', b'QuantizedMklOp')
                helper.set_attr_dtype(quantized_conv_node, "out_type", dtypes.qint32)
                # helper.set_attr_dtype(quantized_conv_node, "alpha", dtypes.quint8)
                helper.set_attr_dtype(quantized_conv_node, "Tbias", dtypes.float32)
                # if self.device == 'gpu' else dtypes.qint32)
                helper.set_attr_string_list(quantized_conv_node, "fused_ops", [b"BiasAdd"])

                helper.set_attr_type_list(
                    quantized_conv_node,
                    "Thost_inputs",
                    [
                        input_data_type.as_datatype_enum,
                        dtypes.qint8.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,  # if self.device == 'gpu' else dtypes.qint32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ],
                )
                helper.set_attr_type_list(
                    quantized_conv_node,
                    "Thost_outputs",
                    [
                        dtypes.qint32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ],
                )

                self.add_output_graph_node(quantized_conv_node)
                quantize_down_name = self._add_quantize_down_nodes(node, quantized_node_name, dtypes.qint8, False)
                self._intel_cpu_add_dequantize_result_node(
                    quantize_down_name, match_node_name[2], dtypes.qint8, performance_only=self.performance_only
                )

            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)

    def apply_newly_conv_single_fusion(self, match_node_name):
        """Apply Conv2D single fusion.

        Dequantize + Conv2D + QuantizeV2
        Dequantize + DepthwiseConv2dNative + QuantizeV2
        """
        skip_node_name = match_node_name[2:]
        matched_node = self.node_name_mapping[match_node_name[1]]
        control_inputs, normal_inputs = self._get_node_input(matched_node.node.name)
        _, q_inputs = self._get_node_input(normal_inputs[0])
        _, q_weights_inputs = self._get_node_input(normal_inputs[1])

        quantizev2_weights_name = q_weights_inputs[0]
        _, weights_name = self._get_node_input(quantizev2_weights_name)
        weights_min_name = weights_name[1]
        weights_max_name = weights_name[2]

        q_weights_name, q_weights_min_name, q_weights_max_name = self._intel_cpu_quantize_weight_eightbit(
            matched_node.node.op, self.node_name_mapping[weights_name[0]].node, self.per_channel
        )

        all_input_names = q_inputs[:1] + [q_weights_name] + q_inputs[1:]
        all_input_names.append(q_weights_min_name)
        all_input_names.append(q_weights_max_name)
        skip_node_name.append(normal_inputs[0])
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
                quantized_node_name = node.name + "_eightbit_quantized_depthwise_conv"
                if node.op == "Conv2D":
                    quantized_node_name = node.name + "_eightbit_quantized_conv"

                node_op = "_FusedQuantizedConv2D" if node.op == "Conv2D" else "_FusedQuantizedDepthwiseConv2D"
                quantized_conv_node = helper.create_node(node_op, quantized_node_name, all_input_names)

                helper.copy_attr(quantized_conv_node, "strides", node.attr["strides"])
                helper.copy_attr(quantized_conv_node, "padding", node.attr["padding"])
                helper.copy_attr(quantized_conv_node, "data_format", node.attr["data_format"])
                if "explicit_paddings" in node.attr:
                    helper.copy_attr(quantized_conv_node, "explicit_paddings", node.attr["explicit_paddings"])
                helper.copy_attr(quantized_conv_node, "dilations", node.attr["dilations"])
                input_data_type = dtypes.quint8 if self._find_relu_node(node) else dtypes.qint8
                helper.set_attr_dtype(quantized_conv_node, "Tinput", input_data_type)
                helper.set_attr_dtype(quantized_conv_node, "Tfilter", dtypes.qint8)
                helper.set_attr_dtype(quantized_conv_node, "Tsummand", dtypes.qint32)
                # helper.set_attr_string(quantized_conv_node, '_kernel', b'QuantizedMklOp')
                helper.set_attr_dtype(quantized_conv_node, "out_type", dtypes.qint32)
                # helper.set_attr_dtype(quantized_conv_node, "alpha", dtypes.quint8)
                helper.set_attr_dtype(quantized_conv_node, "Tbias", dtypes.float32)
                # if self.device == 'gpu' else dtypes.qint32)
                #
                helper.set_attr_string_list(quantized_conv_node, "fused_ops", [])

                helper.set_attr_type_list(
                    quantized_conv_node,
                    "Thost_inputs",
                    [
                        input_data_type.as_datatype_enum,
                        dtypes.qint8.as_datatype_enum,
                        # dtypes.float32.as_datatype_enum if self.device == 'gpu' else dtypes.qint32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ],
                )
                helper.set_attr_type_list(
                    quantized_conv_node,
                    "Thost_outputs",
                    [
                        dtypes.qint32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ],
                )

                self.add_output_graph_node(quantized_conv_node)
                quantize_down_name = self._add_quantize_down_nodes(node, quantized_node_name, dtypes.qint8, False)
                self._intel_cpu_add_dequantize_result_node(
                    quantize_down_name, match_node_name[1], dtypes.qint8, performance_only=self.performance_only
                )

            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)

    def apply_newly_conv_biasadd_addn_relu_fusion(self, match_node_name):
        """Apply Conv2D BiasAdd AddN Relu fusion.

        Dequantize + Conv2D + BiasAdd + AddN + Relu + QuantizeV2
        Dequantize + Conv2D + BiasAdd + AddN + Relu6 + QuantizeV2
        Dequantize + Conv2D + BiasAdd + AddV2 + Relu + QuantizeV2
        Dequantize + Conv2D + BiasAdd + AddV2 + Relu6 + QuantizeV2
        Dequantize + Conv2D + BiasAdd + Add + Relu + QuantizeV2
        Dequantize + Conv2D + BiasAdd + LeakyRelu + AddV2 + QuantizeV2
        Dequantize + Conv2D + BiasAdd + Relu + AddV2(Add) + QuantizeV2
        Dequantize + Conv2D + LeakyRelu + AddV2 + QuantizeV2
        Dequantize + Conv2D + Relu + AddV2(Add) + QuantizeV2
        Dequantize + Conv2D + Add + Add + Relu + QuantizeV2
        Dequantize + Conv2D + BiasAdd + Add + Relu + QuantizeV2
        """
        skip_node_name = match_node_name[2:]
        matched_node = self.node_name_mapping[match_node_name[1]]
        second_node = self.node_name_mapping[match_node_name[2]].node
        need_insert_dummy_biasadd = 1
        add_a_node_name = helper.node_name_from_input(second_node.input[0])
        add_a_node = self.node_name_mapping[add_a_node_name].node
        add_b_node_name = helper.node_name_from_input(second_node.input[1])
        add_b_node = self.node_name_mapping[add_b_node_name].node

        if add_a_node.op != "Const" and add_b_node.op == "Const":
            need_insert_dummy_biasadd = 0
            if len(match_node_name) == 5 and "Relu" in match_node_name[3]:
                return self.apply_newly_conv_biasadd_relu_fusion(match_node_name)

        if need_insert_dummy_biasadd:
            new_match_node_name = self._insert_dummy_biasadd(match_node_name, matched_node)
            # after insert dummy biasadd, that is Conv+dummybiasadd+add*+add*+relu*
            return self.apply_newly_conv_biasadd_addn_fusion(new_match_node_name[:4] + [new_match_node_name[-1]])

        control_inputs, normal_inputs = self._get_node_input(matched_node.node.name)
        _, q_inputs = self._get_node_input(normal_inputs[0])
        _, q_weights_inputs = self._get_node_input(normal_inputs[1])

        quantizev2_weights_name = q_weights_inputs[0]
        _, weights_name = self._get_node_input(quantizev2_weights_name)
        weights_min_name = weights_name[1]
        weights_max_name = weights_name[2]

        third_node = self.node_name_mapping[match_node_name[3]].node
        if third_node.op in ("BiasAdd", "Add", "AddV2", "AddN"):
            sumadd_a_node_name = helper.node_name_from_input(third_node.input[0])
            sumadd_a_node = self.node_name_mapping[sumadd_a_node_name].node
            sumadd_b_node_name = helper.node_name_from_input(third_node.input[1])
            sumadd_b_node = self.node_name_mapping[sumadd_b_node_name].node
            if sumadd_a_node.op != "Const" and sumadd_b_node.op == "Const":
                return self.apply_newly_conv_biasadd_fusion(match_node_name[:3] + [match_node_name[-1]])

        forth_node = self.node_name_mapping[match_node_name[4]].node
        if forth_node.op not in ("LeakyRelu", "Relu"):
            if third_node.op not in ("LeakyRelu", "Relu") and not self._find_relu_node(matched_node.node):
                return self.apply_newly_conv_biasadd_fusion(match_node_name[:3] + [match_node_name[-1]])

        is_leakyrelu_add_fusion = third_node.op == "LeakyRelu" and forth_node.op.find("Add") != -1
        is_relu_add_fusion = third_node.op == "Relu" and forth_node.op.find("Add") != -1

        relu_offset = 0
        if is_leakyrelu_add_fusion or is_relu_add_fusion:
            relu_offset = 1
        sum_index = (
            1
            if match_node_name[2 + relu_offset]
            == self.node_name_mapping[match_node_name[3 + relu_offset]].node.input[0]
            else 0
        )

        sum_node_name = self.node_name_mapping[match_node_name[3 + relu_offset]].node.input[sum_index]
        deq_node = self.node_name_mapping[sum_node_name].node
        if (
            deq_node.op != "LeakyRelu" and deq_node.op != "Dequantize" and deq_node.op != "BiasAdd"
        ) or deq_node.op.find("Quantize") != -1:
            return self.apply_newly_conv_biasadd_fusion(match_node_name[:3] + [match_node_name[-1]])

        q_weights_name, q_weights_min_name, q_weights_max_name = self._intel_cpu_quantize_weight_eightbit(
            matched_node.node.op, self.node_name_mapping[weights_name[0]].node, self.per_channel
        )

        all_input_names = q_inputs[:1] + [q_weights_name] + q_inputs[1:]
        all_input_names.append(q_weights_min_name)
        all_input_names.append(q_weights_max_name)
        skip_node_name.append(normal_inputs[0])
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
                quantized_node_name = node.name + "_eightbit_quantized_depthwise_conv"
                if node.op == "Conv2D":
                    quantized_node_name = node.name + "_eightbit_quantized_conv"

                bias_node_name = self.node_name_mapping[match_node_name[2]].node.input[1]

                if is_leakyrelu_add_fusion or is_relu_add_fusion:
                    relu_node_name = match_node_name[3]
                else:
                    relu_node_name = match_node_name[4]
                is_relu6 = self.node_name_mapping[relu_node_name].node.op == "Relu6"
                is_leakyrelu = self.node_name_mapping[relu_node_name].node.op == "LeakyRelu"

                quantized_node_input_names = (
                    all_input_names[:2] + [bias_node_name] + all_input_names[2:] + [sum_node_name] + control_inputs
                )

                if sum_node_name.find("mul") != -1:
                    quantized_node_input_names = (
                        all_input_names[:2]
                        + [bias_node_name]
                        + [self.node_name_mapping[match_node_name[3 + relu_offset]].node.input[sum_index]]
                        + all_input_names[2:]
                        + control_inputs
                    )

                node_op = "_FusedQuantizedConv2D" if node.op == "Conv2D" else "_FusedQuantizedDepthwiseConv2D"

                quantized_conv_node = helper.create_node(node_op, quantized_node_name, quantized_node_input_names)
                helper.copy_attr(quantized_conv_node, "strides", node.attr["strides"])
                helper.copy_attr(quantized_conv_node, "padding", node.attr["padding"])
                helper.copy_attr(quantized_conv_node, "data_format", node.attr["data_format"])
                if "explicit_paddings" in node.attr:
                    helper.copy_attr(quantized_conv_node, "explicit_paddings", node.attr["explicit_paddings"])
                helper.copy_attr(quantized_conv_node, "dilations", node.attr["dilations"])
                input_data_type = dtypes.quint8 if self._find_relu_node(node) else dtypes.qint8
                helper.set_attr_dtype(quantized_conv_node, "Tinput", input_data_type)
                helper.set_attr_dtype(quantized_conv_node, "Tfilter", dtypes.qint8)
                helper.set_attr_dtype(quantized_conv_node, "out_type", dtypes.qint32)
                if "alpha" in self.node_name_mapping[relu_node_name].node.attr:
                    helper.copy_attr(
                        quantized_conv_node, "alpha", self.node_name_mapping[relu_node_name].node.attr["alpha"]
                    )
                if is_leakyrelu:
                    helper.set_attr_string_list(quantized_conv_node, "fused_ops", [b"BiasAdd", b"Sum", b"LeakyRelu"])
                else:
                    helper.set_attr_string_list(quantized_conv_node, "fused_ops", [b"BiasAdd", b"Sum", b"Relu"])
                helper.set_attr_dtype(quantized_conv_node, "Tbias", dtypes.float32)
                # if self.device == 'gpu' else dtypes.qint32)
                helper.set_attr_dtype(quantized_conv_node, "Tsummand", dtypes.qint32)
                if is_leakyrelu_add_fusion:
                    helper.set_attr_string_list(quantized_conv_node, "fused_ops", [b"BiasAdd", b"LeakyRelu", b"Sum"])
                elif is_relu_add_fusion:
                    helper.set_attr_string_list(quantized_conv_node, "fused_ops", [b"BiasAdd", b"Relu", b"Sum"])

                helper.set_attr_type_list(
                    quantized_conv_node,
                    "Thost_inputs",
                    [
                        input_data_type.as_datatype_enum,
                        dtypes.qint8.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,  # if self.device == 'gpu' else dtypes.qint32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ],
                )
                helper.set_attr_type_list(
                    quantized_conv_node,
                    "Thost_outputs",
                    [
                        dtypes.qint32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ],
                )

                self.add_output_graph_node(quantized_conv_node)

                if is_leakyrelu_add_fusion or is_leakyrelu or is_relu_add_fusion:
                    quantize_down_name = self._add_quantize_down_nodes(node, quantized_node_name, dtypes.qint8, False)
                    self._intel_cpu_add_dequantize_result_node(
                        quantize_down_name,
                        match_node_name[4],
                        dtype=dtypes.qint8,
                        performance_only=self.performance_only,
                    )
                else:
                    dtype = dtypes.quint8
                    if [
                        i
                        for i in self.node_name_mapping[relu_node_name].output
                        if "FusedBatchNorm" in self.node_name_mapping[i].node.op and i in self.op_wise_config_name_list
                    ]:
                        dtype = dtypes.qint8
                    quantize_down_name = self._add_quantize_down_nodes(node, quantized_node_name, dtype, is_relu6)
                    self._intel_cpu_add_dequantize_result_node(
                        quantize_down_name, relu_node_name, dtype, performance_only=self.performance_only
                    )

            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)

    def apply_conv_biasadd_hardswish_fusion(self, match_node_name):
        """Apply Conv2D BiasAdd hardswish fusion.

        Dequantize + Conv2D + BiasAdd + Add + Relu6 + Mul + Mul + QuantizeV2
        Dequantize + Conv2D + Add + Relu6 + Mul + Mul + QuantizeV2
        Dequantize + DepthwiseConv2dNative + BiasAdd + Add + Relu6 + Mul + Mul + QuantizeV2
        Dequantize + DepthwiseConv2dNative + Add + Relu6 + Mul + Mul + QuantizeV2
        """
        skip_node_name = match_node_name[2:]
        matched_node = self.node_name_mapping[match_node_name[1]]
        second_node = self.node_name_mapping[match_node_name[2]].node

        if len(match_node_name) == 7:
            new_match_node_name = self._insert_dummy_biasadd(match_node_name, matched_node)
            return self.apply_conv_biasadd_hardswish_fusion(new_match_node_name)
        need_insert_dummy_biasadd = 1
        add_a_node_name = helper.node_name_from_input(second_node.input[0])
        add_a_node = self.node_name_mapping[add_a_node_name].node
        add_b_node_name = helper.node_name_from_input(second_node.input[1])
        add_b_node = self.node_name_mapping[add_b_node_name].node

        if add_a_node.op != "Const" and add_b_node.op == "Const":
            need_insert_dummy_biasadd = 0
        if need_insert_dummy_biasadd:
            new_match_node_name = self._insert_dummy_biasadd(match_node_name, matched_node)
            # after insert dummy biasadd, that is Conv+dummybiasadd+add+add+relu6+mul+mul
            return self.apply_newly_conv_biasadd_addn_fusion(new_match_node_name[:4] + [new_match_node_name[-1]])

        third_node = self.node_name_mapping[match_node_name[3]].node
        sumadd_a_node_name = helper.node_name_from_input(third_node.input[0])
        sumadd_a_node = self.node_name_mapping[sumadd_a_node_name].node
        sumadd_b_node_name = helper.node_name_from_input(third_node.input[1])
        sumadd_b_node = self.node_name_mapping[sumadd_b_node_name].node
        if sumadd_a_node.op != "Const" and sumadd_b_node.op == "Const":
            need_insert_dummy_biasadd = 0
        else:
            # third node is sumadd
            return self.apply_newly_conv_biasadd_addn_fusion(match_node_name[:4] + [new_match_node_name[-1]])

        control_inputs, normal_inputs = self._get_node_input(matched_node.node.name)
        _, q_inputs = self._get_node_input(normal_inputs[0])
        _, q_weights_inputs = self._get_node_input(normal_inputs[1])
        quantizev2_weights_name = q_weights_inputs[0]

        _, weights_name = self._get_node_input(quantizev2_weights_name)
        weights_min_name = weights_name[1]
        weights_max_name = weights_name[2]

        q_weights_name, q_weights_min_name, q_weights_max_name = self._intel_cpu_quantize_weight_eightbit(
            matched_node.node.op, self.node_name_mapping[weights_name[0]].node, self.per_channel
        )

        all_input_names = q_inputs[:1] + [q_weights_name] + q_inputs[1:]
        all_input_names.append(q_weights_min_name)
        all_input_names.append(q_weights_max_name)
        skip_node_name.append(normal_inputs[0])
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
                quantized_node_name = node.name + "_eightbit_quantized_depthwise_conv"
                if node.op == "Conv2D":
                    quantized_node_name = node.name + "_eightbit_quantized_conv"

                bias_node_name = self.node_name_mapping[match_node_name[2]].node.input[1]
                relu_node_name = match_node_name[4]
                is_relu6 = self.node_name_mapping[relu_node_name].node.op == "Relu6"
                quantized_node_input_names = (
                    all_input_names[:2] + [bias_node_name] + all_input_names[2:] + control_inputs
                )

                node_op = "_FusedQuantizedConv2D" if node.op == "Conv2D" else "_FusedQuantizedDepthwiseConv2D"
                quantized_conv_node = helper.create_node(node_op, quantized_node_name, quantized_node_input_names)

                helper.copy_attr(quantized_conv_node, "strides", node.attr["strides"])
                helper.copy_attr(quantized_conv_node, "padding", node.attr["padding"])
                helper.copy_attr(quantized_conv_node, "data_format", node.attr["data_format"])
                if "alpha" in self.node_name_mapping[relu_node_name].node.attr:
                    helper.copy_attr(
                        quantized_conv_node, "alpha", self.node_name_mapping[relu_node_name].node.attr["alpha"]
                    )
                if "explicit_paddings" in node.attr:
                    helper.copy_attr(quantized_conv_node, "explicit_paddings", node.attr["explicit_paddings"])
                helper.copy_attr(quantized_conv_node, "dilations", node.attr["dilations"])
                input_data_type = dtypes.quint8 if self._find_relu_node(node) else dtypes.qint8
                helper.set_attr_dtype(quantized_conv_node, "Tinput", input_data_type)
                helper.set_attr_dtype(quantized_conv_node, "Tfilter", dtypes.qint8)
                helper.set_attr_dtype(quantized_conv_node, "Tsummand", dtypes.qint32)
                # helper.set_attr_string(quantized_conv_node, '_kernel', b'QuantizedMklOp')
                helper.set_attr_dtype(quantized_conv_node, "out_type", dtypes.qint32)
                # helper.set_attr_dtype(quantized_conv_node, "alpha", dtypes.quint8)
                helper.set_attr_dtype(quantized_conv_node, "Tbias", dtypes.float32)
                # if self.device == 'gpu' else dtypes.qint32)
                helper.set_attr_string_list(quantized_conv_node, "fused_ops", [b"BiasAdd", b"_FusedHardSwish"])
                helper.set_attr_type_list(
                    quantized_conv_node,
                    "Thost_inputs",
                    [
                        input_data_type.as_datatype_enum,
                        dtypes.qint8.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,  # if self.device == 'gpu' else dtypes.qint32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ],
                )
                helper.set_attr_type_list(
                    quantized_conv_node,
                    "Thost_outputs",
                    [
                        dtypes.qint32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ],
                )

                self.add_output_graph_node(quantized_conv_node)
                quantize_down_name = self._add_quantize_down_nodes(node, quantized_node_name, dtypes.quint8, is_relu6)
                self._intel_cpu_add_dequantize_result_node(
                    quantize_down_name, match_node_name[6], dtypes.quint8, performance_only=self.performance_only
                )

            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)

    def apply_newly_conv_biasadd_swishf32_fusion(self, match_node_name):
        """Apply Conv2D BiasAdd swishf32 fusion.

        Dequantize + Conv2D + BiasAdd + swish_f32 + QuantizeV2
        Dequantize + Conv2D + Add + swish_f32 + QuantizeV2
        Dequantize + Conv2D + AddV2 + swish_f32 + QuantizeV2
        Dequantize + Conv2D + swish_f32 + QuantizeV2
        Dequantize + DepthwiseConv2dNative + BiasAdd + swish_f32 + QuantizeV2
        Dequantize + DepthwiseConv2dNative + Add + swish_f32 + QuantizeV2
        Dequantize + DepthwiseConv2dNative + AddV2 + swish_f32 + QuantizeV2
        Dequantize + DepthwiseConv2dNative + swish_f32 + QuantizeV2
        """
        skip_node_name = match_node_name[2:]
        matched_node = self.node_name_mapping[match_node_name[1]]
        second_node = self.node_name_mapping[match_node_name[2]].node

        if second_node.op == "swish_f32":
            new_match_node_name = self._insert_dummy_biasadd(match_node_name, matched_node)
            return self.apply_newly_conv_biasadd_swishf32_fusion(new_match_node_name)

        need_insert_dummy_biasadd = 1
        add_a_node_name = helper.node_name_from_input(second_node.input[0])
        add_a_node = self.node_name_mapping[add_a_node_name].node
        add_b_node_name = helper.node_name_from_input(second_node.input[1])
        add_b_node = self.node_name_mapping[add_b_node_name].node

        if add_a_node.op != "Const" and add_b_node.op == "Const":
            need_insert_dummy_biasadd = 0
        if need_insert_dummy_biasadd:
            new_match_node_name = self._insert_dummy_biasadd(match_node_name, matched_node)
            # TF not support ['BiasAdd', 'Sum', '_FusedSwish'] pattern yet
            return self.apply_newly_conv_biasadd_addn_fusion(new_match_node_name[:4] + [new_match_node_name[-1]])

        control_inputs, normal_inputs = self._get_node_input(matched_node.node.name)
        _, q_inputs = self._get_node_input(normal_inputs[0])
        _, q_weights_inputs = self._get_node_input(normal_inputs[1])
        quantizev2_weights_name = q_weights_inputs[0]

        _, weights_name = self._get_node_input(quantizev2_weights_name)
        weights_min_name = weights_name[1]
        weights_max_name = weights_name[2]

        q_weights_name, q_weights_min_name, q_weights_max_name = self._intel_cpu_quantize_weight_eightbit(
            matched_node.node.op, self.node_name_mapping[weights_name[0]].node, self.per_channel
        )

        all_input_names = q_inputs[:1] + [q_weights_name] + q_inputs[1:]
        all_input_names.append(q_weights_min_name)
        all_input_names.append(q_weights_max_name)
        skip_node_name.append(normal_inputs[0])
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
                quantized_node_name = node.name + "_eightbit_quantized_depthwise_conv"
                if node.op == "Conv2D":
                    quantized_node_name = node.name + "_eightbit_quantized_conv"

                bias_node_name = self.node_name_mapping[match_node_name[2]].node.input[1]
                swish_node_name = match_node_name[3]
                quantized_node_input_names = (
                    all_input_names[:2] + [bias_node_name] + all_input_names[2:] + control_inputs
                )

                node_op = "_FusedQuantizedDepthwiseConv2D"
                if node.op == "Conv2D":
                    node_op = "_FusedQuantizedConv2D"
                quantized_conv_node = helper.create_node(node_op, quantized_node_name, quantized_node_input_names)

                helper.copy_attr(quantized_conv_node, "strides", node.attr["strides"])
                helper.copy_attr(quantized_conv_node, "padding", node.attr["padding"])
                helper.copy_attr(quantized_conv_node, "data_format", node.attr["data_format"])
                if "alpha" in self.node_name_mapping[swish_node_name].node.attr:
                    helper.copy_attr(
                        quantized_conv_node, "alpha", self.node_name_mapping[swish_node_name].node.attr["alpha"]
                    )
                if "explicit_paddings" in node.attr:
                    helper.copy_attr(quantized_conv_node, "explicit_paddings", node.attr["explicit_paddings"])
                helper.copy_attr(quantized_conv_node, "dilations", node.attr["dilations"])
                input_data_type = dtypes.quint8 if self._find_relu_node(node) else dtypes.qint8
                helper.set_attr_dtype(quantized_conv_node, "Tinput", input_data_type)
                helper.set_attr_dtype(quantized_conv_node, "Tfilter", dtypes.qint8)
                helper.set_attr_dtype(quantized_conv_node, "Tsummand", dtypes.qint32)
                helper.set_attr_dtype(quantized_conv_node, "out_type", dtypes.qint32)
                helper.set_attr_dtype(
                    quantized_conv_node, "Tbias", dtypes.float32 if self.device == "gpu" else dtypes.qint32
                )
                fused_ops = [b"BiasAdd", b"_FusedSwish"]
                helper.set_attr_string_list(quantized_conv_node, "fused_ops", fused_ops)
                helper.set_attr_type_list(
                    quantized_conv_node,
                    "Thost_inputs",
                    [
                        input_data_type.as_datatype_enum,
                        dtypes.qint8.as_datatype_enum,
                        dtypes.float32.as_datatype_enum if self.device == "gpu" else dtypes.qint32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ],
                )
                helper.set_attr_type_list(
                    quantized_conv_node,
                    "Thost_outputs",
                    [
                        dtypes.qint32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ],
                )
                self.add_output_graph_node(quantized_conv_node)
                quantize_down_name = self._add_quantize_down_nodes(node, quantized_node_name, dtypes.qint8, False)
                self._intel_cpu_add_dequantize_result_node(
                    quantize_down_name, swish_node_name, dtype=dtypes.qint8, performance_only=self.performance_only
                )
            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)

    def apply_newly_conv_biasadd_addn_fusion(self, match_node_name):
        """Apply Conv2D BiasAdd AddN fusion.

        Dequantize + Conv2D + Add + Add + QuantizeV2
        Dequantize + Conv2D + AddV2 + Add + QuantizeV2
        Dequantize + Conv2D + BiasAdd + Add + QuantizeV2
        """
        skip_node_name = match_node_name[2:]
        matched_node = self.node_name_mapping[match_node_name[1]]

        second_node = self.node_name_mapping[match_node_name[2]].node
        need_insert_dummy_biasadd = 1
        add_a_node_name = helper.node_name_from_input(second_node.input[0])
        add_a_node = self.node_name_mapping[add_a_node_name].node
        add_b_node_name = helper.node_name_from_input(second_node.input[1])
        add_b_node = self.node_name_mapping[add_b_node_name].node

        if add_a_node.op != "Const" and add_b_node.op == "Const":
            need_insert_dummy_biasadd = 0
        if need_insert_dummy_biasadd:
            new_match_node_name = self._insert_dummy_biasadd(match_node_name, matched_node)
            # after insert dummy biasadd, that is Conv+dummybiasadd+add*+add*
            return self.apply_newly_conv_biasadd_addn_fusion(new_match_node_name[:4] + [new_match_node_name[-1]])

        third_node = self.node_name_mapping[match_node_name[3]].node
        sumadd_a_node_name = helper.node_name_from_input(third_node.input[0])
        sumadd_a_node = self.node_name_mapping[sumadd_a_node_name].node
        sumadd_b_node_name = helper.node_name_from_input(third_node.input[1])
        sumadd_b_node = self.node_name_mapping[sumadd_b_node_name].node
        if sumadd_a_node.op != "Const" and sumadd_b_node.op == "Const":
            return self.apply_newly_conv_biasadd_fusion(match_node_name[:3] + [new_match_node_name[-1]])

        sum_index = 1 if match_node_name[2] == self.node_name_mapping[match_node_name[3]].node.input[0] else 0
        sum_node_name = self.node_name_mapping[match_node_name[3]].node.input[sum_index]
        deq_node = self.node_name_mapping[sum_node_name].node
        if deq_node.op != "Dequantize" or deq_node.op.find("Quantize") != -1:
            return self.apply_newly_conv_biasadd_fusion(match_node_name[:3] + [match_node_name[-1]])

        control_inputs, normal_inputs = self._get_node_input(matched_node.node.name)
        _, q_inputs = self._get_node_input(normal_inputs[0])
        _, q_weights_inputs = self._get_node_input(normal_inputs[1])
        quantizev2_weights_name = q_weights_inputs[0]

        _, weights_name = self._get_node_input(quantizev2_weights_name)
        weights_min_name = weights_name[1]
        weights_max_name = weights_name[2]

        q_weights_name, q_weights_min_name, q_weights_max_name = self._intel_cpu_quantize_weight_eightbit(
            matched_node.node.op, self.node_name_mapping[weights_name[0]].node, self.per_channel
        )

        all_input_names = q_inputs[:1] + [q_weights_name] + q_inputs[1:]
        all_input_names.append(q_weights_min_name)
        all_input_names.append(q_weights_max_name)
        skip_node_name.append(normal_inputs[0])
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
                quantized_node_name = node.name + "_eightbit_quantized_depthwise_conv"
                if node.op == "Conv2D":
                    quantized_node_name = node.name + "_eightbit_quantized_conv"
                bias_node_name = self.node_name_mapping[match_node_name[2]].node.input[1]

                quantized_node_input_names = (
                    all_input_names[:2] + [bias_node_name] + all_input_names[2:] + [sum_node_name] + control_inputs
                )
                node_op = "_FusedQuantizedConv2D" if node.op == "Conv2D" else "_FusedQuantizedDepthwiseConv2D"

                quantized_conv_node = helper.create_node(node_op, quantized_node_name, quantized_node_input_names)
                helper.copy_attr(quantized_conv_node, "strides", node.attr["strides"])
                helper.copy_attr(quantized_conv_node, "padding", node.attr["padding"])
                helper.copy_attr(quantized_conv_node, "data_format", node.attr["data_format"])
                if "explicit_paddings" in node.attr:
                    helper.copy_attr(quantized_conv_node, "explicit_paddings", node.attr["explicit_paddings"])
                helper.copy_attr(quantized_conv_node, "dilations", node.attr["dilations"])
                input_data_type = dtypes.quint8 if self._find_relu_node(node) else dtypes.qint8
                helper.set_attr_dtype(quantized_conv_node, "Tinput", input_data_type)
                helper.set_attr_dtype(quantized_conv_node, "Tfilter", dtypes.qint8)
                helper.set_attr_dtype(quantized_conv_node, "out_type", dtypes.qint32)
                helper.set_attr_string_list(quantized_conv_node, "fused_ops", [b"BiasAdd", b"Sum"])
                helper.set_attr_dtype(quantized_conv_node, "Tbias", dtypes.float32)
                # if self.device == 'gpu' else dtypes.qint32)
                helper.set_attr_dtype(quantized_conv_node, "Tsummand", dtypes.qint32)

                helper.set_attr_type_list(
                    quantized_conv_node,
                    "Thost_inputs",
                    [
                        input_data_type.as_datatype_enum,
                        dtypes.qint8.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,  # if self.device == 'gpu' else dtypes.qint32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ],
                )
                helper.set_attr_type_list(
                    quantized_conv_node,
                    "Thost_outputs",
                    [
                        dtypes.qint32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                        dtypes.float32.as_datatype_enum,
                    ],
                )

                self.add_output_graph_node(quantized_conv_node)
                quantize_down_name = self._add_quantize_down_nodes(node, quantized_node_name, dtypes.qint8, False)
                self._intel_cpu_add_dequantize_result_node(
                    quantize_down_name, match_node_name[3], dtype=dtypes.qint8, performance_only=self.performance_only
                )
            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)

    def get_longest_fuse(self):
        """Get the longest fusion pattern."""
        self._get_op_list()

        matched_rule, matched_node_name = self._is_match_conv(self.sorted_patterns)
        return matched_rule, matched_node_name

    def apply_the_transform(self):
        """Quantize Conv and apply the fusion."""
        self._get_op_list()
        matched_rule, matched_node_name = self._is_match_conv(self.sorted_patterns, True)
        if matched_node_name:
            self.output_graph = graph_pb2.GraphDef()
            fusion_name = "".join(matched_rule)
            if fusion_name in self.fusion_mapping:
                if fusion_name.find("DequantizeConv2DAddReluQuantizeV2") != -1:
                    for input_name in self.node_name_mapping[matched_node_name[2]].node.input:
                        input_node_name = helper.node_name_from_input(input_name)
                        if input_node_name != matched_node_name[1]:
                            add_const_input_node = self.node_name_mapping[input_node_name].node
                            add_node_content = tensor_util.MakeNdarray(add_const_input_node.attr["value"].tensor)
                            if add_node_content.ndim != 1:
                                fusion_name = "DequantizeConv2DQuantizeV2"
                                matched_node_name = matched_node_name[:2] + [matched_node_name[-1]]
                self.fusion_mapping[fusion_name](matched_node_name)
            else:  # pragma: no cover
                self.logger.info("Unknown fusion pattern {}.".format(fusion_name))
                if self.remove_redundant_quant_flag:
                    self.input_graph = self.remove_redundant_quantization(self.input_graph)
                return self.input_graph, self.exclude_conv_nodes

            self.input_graph = self.output_graph
            self._reset_output_node_maps()
            if self.remove_redundant_quant_flag:
                self.output_graph = self.remove_redundant_quantization(self.output_graph)

            return self.output_graph, self.exclude_conv_nodes

        if self.remove_redundant_quant_flag:
            self.input_graph = self.remove_redundant_quantization(self.input_graph)
        return self.input_graph, self.exclude_conv_nodes

    def _is_match_conv(self, patterns, qdq_inserted=False):
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

                if ((v in ("Conv2D", "DepthwiseConv2dNative") and not self.enable_s8)) and not self._find_relu_node(
                    cur_node
                ):
                    self.exclude_conv_nodes.append(cur_node.name)
                    continue

                _, normal_inputs = self._get_node_input(cur_node.name)
                if self.node_name_mapping[normal_inputs[1].rsplit(":")[0]].node.op == "Split":
                    self.exclude_conv_nodes.append(cur_node.name)
                    continue

                for sub_rule in patterns:
                    if sub_rule[0] != "Dequantize" or sub_rule[-1] != "QuantizeV2":
                        self.exclude_conv_nodes.append(cur_node.name)
                        continue
                    if v != sub_rule[1]:
                        self.exclude_conv_nodes.append(cur_node.name)
                        continue

                    if qdq_inserted:
                        if (
                            self.node_name_mapping[normal_inputs[0]].node.op != "Dequantize"
                            or self.node_name_mapping[normal_inputs[1]].node.op != "Dequantize"
                        ):
                            continue

                    sub_rule_len = len(sub_rule) - 2
                    check_hardswish = True if sub_rule_len > 4 else False
                    self.logger.debug("Try to apply rule: {}".format(sub_rule))

                    cur_node_name = list(self.node_name_mapping.keys())[k]

                    matched_node_name.clear()
                    matched_node_name.append(sub_rule[0])
                    matched_node_name.append(cur_node_name)

                    count = 0
                    while sub_rule_len > 1:
                        if not self.node_name_mapping[cur_node_name].output:
                            self.logger.debug("Fail to match {}".format(sub_rule))
                            break

                        next_node_name = self.node_name_mapping[cur_node_name].output[0]

                        is_shared_output = True if len(self.node_name_mapping[cur_node_name].output) > 1 else False

                        add_op_quantizable = True
                        is_hardswish = False
                        if is_shared_output:
                            if next_node_name.find("hard_swish") != -1:
                                self.logger.debug("Find Hard Swish pattern ......")
                                is_hardswish = True
                                count = count + 1
                                if next_node_name.find("add") == -1:
                                    next_node_name = self.node_name_mapping[cur_node_name].output[1]
                            else:
                                add_op_quantizable = False
                        next_node_op = self.node_name_mapping[next_node_name].node.op
                        if add_op_quantizable and next_node_op == sub_rule[-sub_rule_len]:
                            if not is_shared_output:
                                matched_node_name.append(next_node_name)
                                sub_rule_len -= 1
                                cur_node_name = next_node_name
                            elif is_hardswish:
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
                        if (
                            check_hardswish
                            and sub_rule[-2] == "Mul"
                            and sub_rule[-3] == "Mul"
                            and sub_rule[-4] == "Relu6"
                            and sub_rule[-5] == "Add"
                            and count != 1
                        ):
                            matched_node_name.clear()
                            self.logger.debug("Fail to match {}.".format(sub_rule))
                            break
                        self.logger.debug("Match {} on nodes {}.".format(sub_rule, matched_node_name))
                        return sub_rule, matched_node_name

        return None, None
