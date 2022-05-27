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

import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util

from .quantize_graph_common import QuantizeGraphHelper as helper
from .quantize_graph_base import QuantizeNodeBase
from ..util import version1_gte_version2
import numpy as np

class FuseNodeStartWithConv2d(QuantizeNodeBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sorted_patterns = sorted(self.patterns,
                                      key=lambda i: len(i),
                                      reverse=True)
        self.use_new_api = version1_gte_version2(tf.version.VERSION, '2.8.0')
        if self.use_new_api:
            self.fusion_mapping = {
                'Conv2DBiasAdd': self.apply_newly_conv_biasadd_fusion,
                'Conv2DBiasAddAddNRelu': self.apply_newly_conv_biasadd_addn_relu_fusion,
                'Conv2DBiasAddAddNRelu6': self.apply_newly_conv_biasadd_addn_relu_fusion,
                'Conv2DBiasAddAddV2Relu': self.apply_newly_conv_biasadd_addn_relu_fusion,
                'Conv2DBiasAddAddV2Relu6': self.apply_newly_conv_biasadd_addn_relu_fusion,
                'Conv2DBiasAddAddRelu': self.apply_newly_conv_biasadd_addn_relu_fusion,
                'Conv2DBiasAddRelu6': self.apply_newly_conv_biasadd_relu_fusion,
                'Conv2DBiasAddRelu': self.apply_newly_conv_biasadd_relu_fusion,
                'Conv2DBiasAddElu': self.apply_newly_conv_biasadd_relu_fusion,
                'Conv2DBiasAddLeakyRelu': self.apply_newly_conv_biasadd_relu_fusion,
                'Conv2DBiasAddLeakyReluAddV2': self.apply_newly_conv_biasadd_addn_relu_fusion,
                'Conv2DAddRelu6': self.apply_newly_conv_biasadd_relu_fusion,
                'Conv2DAddRelu': self.apply_newly_conv_biasadd_relu_fusion,
                'Conv2DBiasAddAddRelu6MulMul': self.apply_conv_biasadd_hardswish_fusion,
                'Conv2DBiasAddswish_f32': self.apply_newly_conv_biasadd_swishf32_fusion,
                'Conv2DAddswish_f32': self.apply_newly_conv_biasadd_swishf32_fusion,
                'Conv2DAddV2swish_f32': self.apply_newly_conv_biasadd_swishf32_fusion,
                'DepthwiseConv2dNativeBiasAddAddRelu6MulMul': self.apply_conv_biasadd_hardswish_fusion,
                'DepthwiseConv2dNativeBiasAddswish_f32': self.apply_newly_conv_biasadd_swishf32_fusion,
                'DepthwiseConv2dNativeAddswish_f32': self.apply_newly_conv_biasadd_swishf32_fusion,
                'DepthwiseConv2dNativeAddV2swish_f32': self.apply_newly_conv_biasadd_swishf32_fusion,
                'DepthwiseConv2dNativeAddRelu6':
                self.apply_newly_conv_biasadd_relu_fusion,
                'DepthwiseConv2dNativeBiasAddRelu':
                self.apply_newly_conv_biasadd_relu_fusion,
                'DepthwiseConv2dNativeBiasAdd': self.apply_newly_conv_biasadd_fusion,
                'DepthwiseConv2dNativeBiasAddLeakyRelu': self.apply_newly_conv_biasadd_relu_fusion,
                'DepthwiseConv2dNativeBiasAddRelu6':
                self.apply_newly_conv_biasadd_relu_fusion,
                'Conv2D': self.apply_newly_conv_single_fusion,
                'Conv3D': self.apply_conv3d_single_fusion,
                'Conv3DAdd': self.apply_conv3d_add_fusion,
                'Conv3DAddRelu': self.apply_conv3d_add_relu_fusion,
                'Conv3DAddV2': self.apply_conv3d_add_fusion,
                'Conv3DAddV2Relu': self.apply_conv3d_add_relu_fusion,
                'DepthwiseConv2dNative': self.apply_newly_conv_single_fusion
            }
        else:
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
                'DepthwiseConv2dNativeBiasAddRelu':
                self.apply_conv_biasadd_relu_fusion,
                'DepthwiseConv2dNativeBiasAdd': self.apply_conv_biasadd_fusion,
                'DepthwiseConv2dNativeBiasAddRelu6':
                self.apply_conv_biasadd_relu_fusion,
                'Conv2D': self.apply_conv_single_fusion,
                'DepthwiseConv2dNative': self.apply_conv_single_fusion
            }
    
    def apply_conv3d_add_relu_fusion(self, match_node_name):
        skip_node_name = match_node_name[1:]
        matched_node = self.node_name_mapping[match_node_name[0]]
        control_inputs, normal_inputs = self._get_node_input(matched_node.node.name)
        weight_name = normal_inputs[1]

        third_node = self.node_name_mapping[match_node_name[2]].node 
        if third_node.op != 'LeakyRelu' and not self._find_relu_node(matched_node.node):
            return self.apply_conv3d_add_fusion(match_node_name[:2])

        add_node = self.node_name_mapping[match_node_name[1]].node
        original_add_input = self.node_name_mapping[add_node.input[1]].node
        if original_add_input.op == 'Const':
            shape = tensor_util.MakeNdarray(original_add_input.attr["value"].tensor)
            if shape.ndim > 1 and shape.shape[:-1] == (1,1,1,1):
                squeezed_value = np.squeeze(shape)
                squeezed_node = helper.create_constant_node(match_node_name[1] +'_squeezed', squeezed_value, dtypes.float32)
                skip_node_name.append(add_node.input[1])
                add_node.input[1] = squeezed_node.name
                self.add_output_graph_node(squeezed_node)

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
                # actually DepthwiseConv3dNative is not supported in intel-tf 2.8 yet
                postfix = "_eightbit_quantized_depthwise_conv3d"
                if node.op == "Conv3D":
                    postfix = "_eightbit_quantized_conv3d"
                quantized_node_name = node.name + postfix

                bias_node_name = self.node_name_mapping[match_node_name[1]].node.input[1]
                relu_node_name = match_node_name[2]
                is_relu6 = self.node_name_mapping[relu_node_name].node.op == "Relu6"
                quantized_node_input_names = all_input_names[:2] + \
                    [bias_node_name] + all_input_names[2:] + control_inputs

                is_leakyrelu = self.node_name_mapping[relu_node_name].node.op == "LeakyRelu"

                quantized_conv_node = helper.create_node(
                    "_QuantizedConv3D",
                    quantized_node_name,
                    quantized_node_input_names)

                helper.copy_attr(quantized_conv_node, "strides", node.attr["strides"])
                helper.copy_attr(quantized_conv_node, "padding", node.attr["padding"])
                if "alpha" in self.node_name_mapping[relu_node_name].node.attr:
                    helper.copy_attr(quantized_conv_node, "alpha",
                    self.node_name_mapping[relu_node_name].node.attr["alpha"])
                if node.op != 'DepthwiseConv3dNative' and "padding_list" in node.attr:
                    helper.copy_attr(quantized_conv_node, "padding_list",
                    node.attr["padding_list"])
                helper.copy_attr(quantized_conv_node, "dilations", node.attr["dilations"])
                input_data_type = dtypes.quint8 if self._find_relu_node(node) else dtypes.qint8
                helper.set_attr_dtype(quantized_conv_node, "Tinput", input_data_type)
                helper.set_attr_dtype(quantized_conv_node, "Tfilter",dtypes.qint8)
                helper.set_attr_dtype(quantized_conv_node, "Tsummand", dtypes.qint32)
                # helper.set_attr_string(quantized_conv_node, '_kernel', b'QuantizedMklOp')
                helper.set_attr_dtype(quantized_conv_node, "out_type", dtypes.qint32)
                # helper.set_attr_dtype(quantized_conv_node, "alpha", dtypes.quint8)
                helper.set_attr_dtype(quantized_conv_node, "Tbias", dtypes.float32)
                                                # if self.device == 'gpu' else dtypes.qint32)
                helper.set_attr_string_list(quantized_conv_node, 'fused_ops', [b'BiasAdd', b'Relu'])
                helper.set_attr_type_list(quantized_conv_node, 'input_types', [
                    input_data_type.as_datatype_enum,
                    dtypes.qint8.as_datatype_enum,
                    dtypes.float32.as_datatype_enum,# if self.device == 'gpu' else dtypes.qint32.as_datatype_enum,
                    dtypes.float32.as_datatype_enum,
                    dtypes.float32.as_datatype_enum,
                    dtypes.float32.as_datatype_enum,
                    dtypes.float32.as_datatype_enum,
                 ])
                helper.set_attr_type_list(quantized_conv_node, 'out_types', [
                                          dtypes.qint32.as_datatype_enum,
                                          dtypes.float32.as_datatype_enum,
                                          dtypes.float32.as_datatype_enum, ])

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

    def apply_conv3d_add_fusion(self, match_node_name):
        skip_node_name = match_node_name[1:]
        matched_node = self.node_name_mapping[match_node_name[0]]
        _, normal_inputs = self._get_node_input(matched_node.node.name)
        weight_name = normal_inputs[1]

        add_node = self.node_name_mapping[match_node_name[1]].node
        original_add_input = self.node_name_mapping[add_node.input[1]].node
        if original_add_input.op == 'Const':
            shape = tensor_util.MakeNdarray(original_add_input.attr["value"].tensor)
            if shape.ndim > 1 and shape.shape[:-1] == (1,1,1,1):
                squeezed_value = np.squeeze(shape)
                squeezed_node = helper.create_constant_node(match_node_name[1] +'_squeezed', squeezed_value, dtypes.float32)
                skip_node_name.append(add_node.input[1])
                add_node.input[1] = squeezed_node.name
                self.add_output_graph_node(squeezed_node)


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
                # actually DepthwiseConv3dNative is not supported in intel-tf 2.8 yet
                postfix = "_eightbit_quantized_depthwise_conv3d"
                if node.op == "Conv3D":
                    postfix = "_eightbit_quantized_conv3d"
                quantized_node_name = node.name + postfix
    
                bias_node_name = self.node_name_mapping[match_node_name[1]].node.input[1]
                quantized_node_input_names = all_input_names[:2] + [bias_node_name] + all_input_names[2:]
                if node.op == "Conv3D":
                    quantized_conv_node = helper.create_node(
                        "_QuantizedConv3D",
                        quantized_node_name,
                        quantized_node_input_names)

                helper.copy_attr(quantized_conv_node, "strides", node.attr["strides"])
                helper.copy_attr(quantized_conv_node, "padding", node.attr["padding"])
                if node.op != 'DepthwiseConv3dNative' and "padding_list" in node.attr:
                    helper.copy_attr(quantized_conv_node, "padding_list",node.attr["padding_list"])
                helper.copy_attr(quantized_conv_node, "dilations", node.attr["dilations"])
                input_data_type = dtypes.quint8 if self._find_relu_node(node) else dtypes.qint8
                helper.set_attr_dtype(quantized_conv_node, "Tinput", input_data_type)
                helper.set_attr_dtype(quantized_conv_node, "Tfilter", dtypes.qint8)
                helper.set_attr_dtype(quantized_conv_node, "Tbias", dtypes.float32)
                                                 # if self.device == 'gpu' else dtypes.qint32)
                helper.set_attr_dtype(quantized_conv_node, "Tsummand", dtypes.qint32)
                # helper.set_attr_string(quantized_conv_node, '_kernel', b'QuantizedMklOp')
                helper.set_attr_dtype(quantized_conv_node, "out_type",
                                      dtypes.qint32)
                helper.set_attr_string_list(quantized_conv_node, 'fused_ops', [b'BiasAdd'])

                helper.set_attr_type_list(quantized_conv_node, 'input_types', [
                    input_data_type.as_datatype_enum,
                    dtypes.qint8.as_datatype_enum,
                    dtypes.float32.as_datatype_enum, # if self.device == 'gpu' else dtypes.qint32.as_datatype_enum,
                    dtypes.float32.as_datatype_enum,
                    dtypes.float32.as_datatype_enum,
                    dtypes.float32.as_datatype_enum,
                    dtypes.float32.as_datatype_enum,
                 ])
                helper.set_attr_type_list(quantized_conv_node, 'out_types', [
                                          dtypes.qint32.as_datatype_enum,
                                          dtypes.float32.as_datatype_enum,
                                          dtypes.float32.as_datatype_enum, ])

                self.add_output_graph_node(quantized_conv_node)
                
                quantize_down_name = self._add_quantize_down_nodes(
                    node, quantized_node_name, dtypes.qint8)
                
                self._intel_cpu_add_dequantize_result_node(
                    quantize_down_name, match_node_name[1], dtypes.qint8)
            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)

    def apply_conv3d_single_fusion(self, match_node_name):
        skip_node_name = match_node_name[1:]
        matched_node = self.node_name_mapping[match_node_name[0]]
        _, normal_inputs = self._get_node_input(matched_node.node.name)
        weight_name = normal_inputs[1]

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
                # actually DepthwiseConv3dNative is not supported in intel-tf 2.8 yet
                postfix = "_eightbit_quantized_depthwise_conv3d"
                if node.op == "Conv3D":
                    postfix = "_eightbit_quantized_conv3d"
                quantized_node_name = node.name + postfix
                quantized_conv_node = helper.create_node(
                    "_QuantizedConv3D"
                    if self.per_channel else "_QuantizedConv3D",
                    quantized_node_name, all_input_names)

                helper.copy_attr(quantized_conv_node, "strides", node.attr["strides"])
                helper.copy_attr(quantized_conv_node, "padding", node.attr["padding"])
                if node.op != 'DepthwiseConv3dNative' and "padding_list" in node.attr:
                    helper.copy_attr(quantized_conv_node, "padding_list",
                                     node.attr["padding_list"])
                helper.copy_attr(quantized_conv_node, "dilations", node.attr["dilations"])
                input_data_type = dtypes.quint8 if self._find_relu_node(node) else dtypes.qint8
                helper.set_attr_dtype(quantized_conv_node, "Tinput", input_data_type)
                helper.set_attr_dtype(quantized_conv_node, "Tfilter", dtypes.qint8)
                helper.set_attr_dtype(quantized_conv_node, "out_type", dtypes.qint32)
                helper.set_attr_dtype(quantized_conv_node, "Tsummand", dtypes.qint32)
                helper.set_attr_dtype(quantized_conv_node, "Tbias", dtypes.float32)
                                                # if self.device == 'gpu' else dtypes.qint32)
                # helper.set_attr_string(quantized_conv_node, '_kernel', b'QuantizedMklOp')
                helper.set_attr_string_list(quantized_conv_node, 'fused_ops', [])

                helper.set_attr_type_list(quantized_conv_node, 'input_types', [
                    input_data_type.as_datatype_enum,
                    dtypes.qint8.as_datatype_enum,
                    #dtypes.float32.as_datatype_enum if self.device == 'gpu' else dtypes.qint32.as_datatype_enum,
                    dtypes.float32.as_datatype_enum,
                    dtypes.float32.as_datatype_enum,
                    dtypes.float32.as_datatype_enum,
                    dtypes.float32.as_datatype_enum,
                 ])
                helper.set_attr_type_list(quantized_conv_node, 'out_types', [
                                          dtypes.qint32.as_datatype_enum,
                                          dtypes.float32.as_datatype_enum,
                                          dtypes.float32.as_datatype_enum,])
                self.add_output_graph_node(quantized_conv_node)
                quantize_down_name = self._add_quantize_down_nodes(
                    node, quantized_node_name, dtypes.qint8)
                self._intel_cpu_add_dequantize_result_node(
                    quantize_down_name, node.name, dtypes.qint8)
            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)


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
                self.logger.debug("Skip node {}.".format(node.name))
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
                self.logger.debug("Skip node {}.".format(node.name))
            elif node.name == match_node_name[0]:

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
                if "padding_list" in node.attr:
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
                self.logger.debug("Matched node {} with input {}.".format(node.name, node.input))

                quantized_node_name = node.name + "_eightbit_quantized_conv"
                if node.op == "DepthwiseConv2dNative":
                    quantized_node_name = node.name + "_eightbit_quantized_depthwise_conv"

                bias_node_name = self.node_name_mapping[
                    match_node_name[1]].node.input[1]
                quantized_node_input_names = all_input_names[:2] + [
                    bias_node_name
                ] + all_input_names[2:] + control_inputs

                quantized_conv_node = helper.create_node(
                    "QuantizedConv2DWithBias" if node.op == 'Conv2D' \
                        else 'QuantizedDepthwiseConv2DWithBias',
                    quantized_node_name,
                    quantized_node_input_names)

                helper.copy_attr(quantized_conv_node, "strides", node.attr["strides"])
                helper.copy_attr(quantized_conv_node, "padding", node.attr["padding"])
                if "padding_list" in node.attr:
                    helper.copy_attr(quantized_conv_node, "padding_list", node.attr["padding_list"])
                helper.copy_attr(quantized_conv_node, "dilations", node.attr["dilations"])

                input_data_type = dtypes.quint8 if self._find_relu_node(node) else dtypes.qint8

                helper.set_attr_dtype(quantized_conv_node, "Tinput", input_data_type)
                helper.set_attr_dtype(quantized_conv_node, "Tfilter", dtypes.qint8)
                helper.set_attr_dtype(quantized_conv_node, "out_type", dtypes.qint32)
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
        if third_node.op != 'LeakyRelu' and not self._find_relu_node(matched_node.node):
            return self.apply_conv_biasadd_fusion(match_node_name[:2])

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
                self.logger.debug("Matched node {} with input {}.".format(node.name, node.input))

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


    def apply_newly_conv_biasadd_relu_fusion(self, match_node_name):
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
                self.logger.debug("skip node {}".format(node.name))
            elif node.name == match_node_name[0]:
                self.logger.debug("Matched node {} with input {}.".format(node.name, node.input))
                quantized_node_name = node.name + "_eightbit_quantized_depthwise_conv"
                if node.op == "Conv2D":
                    quantized_node_name = node.name + "_eightbit_quantized_conv"

                bias_node_name = self.node_name_mapping[match_node_name[1]].node.input[1]
                relu_node_name = match_node_name[2]
                is_relu6 = self.node_name_mapping[relu_node_name].node.op == "Relu6"
                quantized_node_input_names = all_input_names[:2] + \
                    [bias_node_name] + all_input_names[2:] + control_inputs
                is_leakyrelu = self.node_name_mapping[relu_node_name].node.op == "LeakyRelu"
                is_elu = self.node_name_mapping[relu_node_name].node.op == "Elu"
                
                node_op = '_QuantizedDepthwiseConv2D' 
                if node.op == 'Conv2D':
                    node_op = "_QuantizedConv2D"
                quantized_conv_node = helper.create_node(node_op, quantized_node_name,
                    quantized_node_input_names)
 
                helper.copy_attr(quantized_conv_node, "strides", node.attr["strides"])
                helper.copy_attr(quantized_conv_node, "padding", node.attr["padding"])
                if "alpha" in self.node_name_mapping[relu_node_name].node.attr:
                    helper.copy_attr(quantized_conv_node, "alpha",
                    self.node_name_mapping[relu_node_name].node.attr["alpha"])
                if "padding_list" in node.attr:
                    helper.copy_attr(quantized_conv_node, "padding_list",
                    node.attr["padding_list"])
                helper.copy_attr(quantized_conv_node, "dilations", node.attr["dilations"])
                input_data_type = dtypes.quint8 if self._find_relu_node(node) else dtypes.qint8
                helper.set_attr_dtype(quantized_conv_node, "Tinput", input_data_type)
                helper.set_attr_dtype(quantized_conv_node, "Tfilter",dtypes.qint8)
                helper.set_attr_dtype(quantized_conv_node, "Tsummand", dtypes.qint32)
                # helper.set_attr_string(quantized_conv_node, '_kernel', b'QuantizedMklOp')
                helper.set_attr_dtype(quantized_conv_node, "out_type", dtypes.qint32)
                # helper.set_attr_dtype(quantized_conv_node, "alpha", dtypes.quint8)
                helper.set_attr_dtype(quantized_conv_node, "Tbias", dtypes.float32)
                                                #if self.device == 'gpu' else dtypes.qint32)
                fused_ops = [b'BiasAdd', b'Relu']
                if is_leakyrelu:
                    fused_ops = [b'BiasAdd', b'LeakyRelu']
                if is_elu:
                    fused_ops = [b'BiasAdd', b'Elu'] 
                helper.set_attr_string_list(quantized_conv_node, 'fused_ops', fused_ops)
                helper.set_attr_type_list(quantized_conv_node, 'input_types', [
                    input_data_type.as_datatype_enum,
                    dtypes.qint8.as_datatype_enum,
                    dtypes.float32.as_datatype_enum,# if self.device == 'gpu' else dtypes.qint32.as_datatype_enum,
                    dtypes.float32.as_datatype_enum,
                    dtypes.float32.as_datatype_enum,
                    dtypes.float32.as_datatype_enum,
                    dtypes.float32.as_datatype_enum,
                 ])
                helper.set_attr_type_list(quantized_conv_node, 'out_types', [
                                          dtypes.qint32.as_datatype_enum,
                                          dtypes.float32.as_datatype_enum,
                                          dtypes.float32.as_datatype_enum, ])
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


    def apply_newly_conv_biasadd_fusion(self, match_node_name):
        skip_node_name = match_node_name[1:]
        matched_node = self.node_name_mapping[match_node_name[0]]
        control_inputs, normal_inputs = self._get_node_input(
            matched_node.node.name)
        weight_name = normal_inputs[1]
        #if not self._find_relu_node(matched_node.node):
        #    return self.apply_conv_biasadd_fusion(match_node_name[:2])
 
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
                self.logger.debug("Matched node {} with input {}.".format(node.name, node.input))
                quantized_node_name = node.name + "_eightbit_quantized_depthwise_conv"
                if node.op == "Conv2D":
                    quantized_node_name = node.name + "_eightbit_quantized_conv"

                bias_node_name = self.node_name_mapping[match_node_name[1]].node.input[1]
                quantized_node_input_names = all_input_names[:2] + \
                    [bias_node_name] + all_input_names[2:] + control_inputs
 
                node_op = "_QuantizedConv2D" if node.op == 'Conv2D' \
                        else '_QuantizedDepthwiseConv2D'
                quantized_conv_node = helper.create_node(node_op, quantized_node_name,
                    quantized_node_input_names)
 
                helper.copy_attr(quantized_conv_node, "strides", node.attr["strides"])
                helper.copy_attr(quantized_conv_node, "padding", node.attr["padding"])
                if "padding_list" in node.attr:
                    helper.copy_attr(quantized_conv_node, "padding_list",
                    node.attr["padding_list"])
                helper.copy_attr(quantized_conv_node, "dilations", node.attr["dilations"])
                input_data_type = dtypes.quint8 if self._find_relu_node(node) else dtypes.qint8
                helper.set_attr_dtype(quantized_conv_node, "Tinput", input_data_type)
                helper.set_attr_dtype(quantized_conv_node, "Tfilter",dtypes.qint8)
                helper.set_attr_dtype(quantized_conv_node, "Tsummand", dtypes.qint32)
                # helper.set_attr_string(quantized_conv_node, '_kernel', b'QuantizedMklOp')
                helper.set_attr_dtype(quantized_conv_node, "out_type", dtypes.qint32)
                # helper.set_attr_dtype(quantized_conv_node, "alpha", dtypes.quint8)
                helper.set_attr_dtype(quantized_conv_node, "Tbias", dtypes.float32)
                                                #if self.device == 'gpu' else dtypes.qint32)
                helper.set_attr_string_list(quantized_conv_node, 'fused_ops', [b'BiasAdd'])

                helper.set_attr_type_list(quantized_conv_node, 'input_types', [
                    input_data_type.as_datatype_enum,
                    dtypes.qint8.as_datatype_enum,
                    dtypes.float32.as_datatype_enum,# if self.device == 'gpu' else dtypes.qint32.as_datatype_enum,
                    dtypes.float32.as_datatype_enum,
                    dtypes.float32.as_datatype_enum,
                    dtypes.float32.as_datatype_enum,
                    dtypes.float32.as_datatype_enum,
                 ])
                helper.set_attr_type_list(quantized_conv_node, 'out_types', [
                                          dtypes.qint32.as_datatype_enum,
                                          dtypes.float32.as_datatype_enum,
                                          dtypes.float32.as_datatype_enum, ])
 
                self.add_output_graph_node(quantized_conv_node)
                quantize_down_name = self._add_quantize_down_nodes(
                    node, quantized_node_name, dtypes.qint8, False)
                self._intel_cpu_add_dequantize_result_node(
                    quantize_down_name, match_node_name[1], dtypes.qint8)
 
            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)

    def apply_newly_conv_single_fusion(self, match_node_name):
        skip_node_name = match_node_name[1:]
        matched_node = self.node_name_mapping[match_node_name[0]]
        _, normal_inputs = self._get_node_input(matched_node.node.name)
        weight_name = normal_inputs[1]
 
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
                self.logger.debug("Matched node {} with input {}.".format(node.name, node.input))
                quantized_node_name = node.name + "_eightbit_quantized_depthwise_conv"
                if node.op == "Conv2D":
                    quantized_node_name = node.name + "_eightbit_quantized_conv"
                    
                node_op = "_QuantizedConv2D" if node.op == 'Conv2D' \
                        else '_QuantizedDepthwiseConv2D'
                quantized_conv_node = helper.create_node(node_op, quantized_node_name,
                    all_input_names)
 
                helper.copy_attr(quantized_conv_node, "strides", node.attr["strides"])
                helper.copy_attr(quantized_conv_node, "padding", node.attr["padding"])
                if "padding_list" in node.attr:
                    helper.copy_attr(quantized_conv_node, "padding_list",
                    node.attr["padding_list"])
                helper.copy_attr(quantized_conv_node, "dilations", node.attr["dilations"])
                input_data_type = dtypes.quint8 if self._find_relu_node(node) else dtypes.qint8
                helper.set_attr_dtype(quantized_conv_node, "Tinput", input_data_type)
                helper.set_attr_dtype(quantized_conv_node, "Tfilter",dtypes.qint8)
                helper.set_attr_dtype(quantized_conv_node, "Tsummand", dtypes.qint32)
                # helper.set_attr_string(quantized_conv_node, '_kernel', b'QuantizedMklOp')
                helper.set_attr_dtype(quantized_conv_node, "out_type", dtypes.qint32)
                # helper.set_attr_dtype(quantized_conv_node, "alpha", dtypes.quint8)
                helper.set_attr_dtype(quantized_conv_node, "Tbias", dtypes.float32)
                                                #if self.device == 'gpu' else dtypes.qint32)
                # 
                helper.set_attr_string_list(quantized_conv_node, 'fused_ops', [])

                helper.set_attr_type_list(quantized_conv_node, 'input_types', [
                    input_data_type.as_datatype_enum,
                    dtypes.qint8.as_datatype_enum,
                    #dtypes.float32.as_datatype_enum if self.device == 'gpu' else dtypes.qint32.as_datatype_enum,
                    dtypes.float32.as_datatype_enum,
                    dtypes.float32.as_datatype_enum,
                    dtypes.float32.as_datatype_enum,
                    dtypes.float32.as_datatype_enum,
                 ])
                helper.set_attr_type_list(quantized_conv_node, 'out_types', [
                                          dtypes.qint32.as_datatype_enum,
                                          dtypes.float32.as_datatype_enum,
                                          dtypes.float32.as_datatype_enum, ])
 
                self.add_output_graph_node(quantized_conv_node)
                quantize_down_name = self._add_quantize_down_nodes(
                    node, quantized_node_name, dtypes.qint8, False)
                self._intel_cpu_add_dequantize_result_node(
                    quantize_down_name, match_node_name[0], dtypes.qint8)
 
            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)

    def apply_newly_conv_biasadd_addn_relu_fusion(self, match_node_name):
        skip_node_name = match_node_name[1:]
        matched_node = self.node_name_mapping[match_node_name[0]]
        control_inputs, normal_inputs = self._get_node_input(
            matched_node.node.name)
        weight_name = normal_inputs[1]

        third_node = self.node_name_mapping[match_node_name[2]].node
        forth_node = self.node_name_mapping[match_node_name[3]].node
        if third_node.op != 'LeakyRelu' and not self._find_relu_node(matched_node.node):
            return self.apply_newly_conv_biasadd_fusion(match_node_name[:2])

        is_leakyrelu_add_fusion = third_node.op == 'LeakyRelu' and forth_node.op.find('Add') != -1
        is_relu_add_fusion = third_node.op == 'Relu' and forth_node.op.find('Add') != -1

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
                self.logger.debug("Matched node {} with input {}.".format(node.name, node.input))
                quantized_node_name = node.name + "_eightbit_quantized_conv"
                bias_node_name = self.node_name_mapping[match_node_name[1]].node.input[1]
                relu_offset = 0
                if is_leakyrelu_add_fusion or is_relu_add_fusion:
                    relu_offset = 1
                if is_leakyrelu_add_fusion or is_relu_add_fusion:
                    relu_node_name = match_node_name[2]
                else:
                    relu_node_name = match_node_name[3]
                is_relu6 = self.node_name_mapping[relu_node_name].node.op == "Relu6"

                sum_index = 1 if match_node_name[1 + relu_offset] == self.node_name_mapping[
                    match_node_name[2 + relu_offset]].node.input[0] else 0

                sum_node_name = self.node_name_mapping[match_node_name[2 + relu_offset]].node.input[sum_index]
                quantized_node_input_names = all_input_names[:2] + [
                    bias_node_name
                ] + all_input_names[2:] + [
                    sum_node_name
                ] + control_inputs
                
                if sum_node_name.find('mul') != -1:
                    quantized_node_input_names = all_input_names[:2] + [
                        bias_node_name
                    ] + [
                        self.node_name_mapping[
                            match_node_name[2 + relu_offset]].node.input[sum_index]
                    ] + all_input_names[2:] + control_inputs

                node_op = "_QuantizedConv2D"

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
                helper.set_attr_string_list(quantized_conv_node, 'fused_ops', [b'BiasAdd', b'Sum', b'Relu'])
                helper.set_attr_dtype(quantized_conv_node, "Tbias", dtypes.float32)
                                                # if self.device == 'gpu' else dtypes.qint32)
                helper.set_attr_dtype(quantized_conv_node, "Tsummand", dtypes.float32)
                if is_leakyrelu_add_fusion:
                    helper.set_attr_string_list(quantized_conv_node, 'fused_ops', [b'BiasAdd', b'LeakyRelu', b'Sum'])
                elif is_relu_add_fusion:
                    helper.set_attr_string_list(quantized_conv_node, 'fused_ops', [b'BiasAdd', b'Relu', b'Sum'])

                helper.set_attr_type_list(quantized_conv_node, 'input_types', [
                    input_data_type.as_datatype_enum,
                    dtypes.qint8.as_datatype_enum,
                    dtypes.float32.as_datatype_enum,# if self.device == 'gpu' else dtypes.qint32.as_datatype_enum,
                    dtypes.float32.as_datatype_enum,
                    dtypes.float32.as_datatype_enum,
                    dtypes.float32.as_datatype_enum,
                    dtypes.float32.as_datatype_enum,
                    dtypes.float32.as_datatype_enum,
                 ])
                helper.set_attr_type_list(quantized_conv_node, 'out_types', [
                                          dtypes.qint32.as_datatype_enum,
                                          dtypes.float32.as_datatype_enum,
                                          dtypes.float32.as_datatype_enum, ])

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
                        quantize_down_name, relu_node_name, dtypes.quint8)

            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)

    def apply_conv_biasadd_hardswish_fusion(self, match_node_name):
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
                self.logger.debug("skip node {}".format(node.name))
            elif node.name == match_node_name[0]:
                self.logger.debug("Matched node {} with input {}.".format(node.name, node.input))
                quantized_node_name = node.name + "_eightbit_quantized_depthwise_conv"
                if node.op == "Conv2D":
                    quantized_node_name = node.name + "_eightbit_quantized_conv"

                bias_node_name = self.node_name_mapping[match_node_name[1]].node.input[1]
                relu_node_name = match_node_name[3]
                is_relu6 = self.node_name_mapping[relu_node_name].node.op == "Relu6"
                quantized_node_input_names = all_input_names[:2] + \
                    [bias_node_name] + all_input_names[2:] + control_inputs

                node_op = "_QuantizedConv2D" if node.op == 'Conv2D' \
                        else '_QuantizedDepthwiseConv2D' 
                quantized_conv_node = helper.create_node(node_op, quantized_node_name,
                    quantized_node_input_names)

                helper.copy_attr(quantized_conv_node, "strides", node.attr["strides"])
                helper.copy_attr(quantized_conv_node, "padding", node.attr["padding"])
                if "alpha" in self.node_name_mapping[relu_node_name].node.attr:
                    helper.copy_attr(quantized_conv_node, "alpha",
                    self.node_name_mapping[relu_node_name].node.attr["alpha"])
                if "padding_list" in node.attr:
                    helper.copy_attr(quantized_conv_node, "padding_list",
                    node.attr["padding_list"])
                helper.copy_attr(quantized_conv_node, "dilations", node.attr["dilations"])
                input_data_type = dtypes.quint8 if self._find_relu_node(node) else dtypes.qint8
                helper.set_attr_dtype(quantized_conv_node, "Tinput", input_data_type)
                helper.set_attr_dtype(quantized_conv_node, "Tfilter",dtypes.qint8)
                helper.set_attr_dtype(quantized_conv_node, "Tsummand", dtypes.qint32)
                # helper.set_attr_string(quantized_conv_node, '_kernel', b'QuantizedMklOp')
                helper.set_attr_dtype(quantized_conv_node, "out_type", dtypes.qint32)
                # helper.set_attr_dtype(quantized_conv_node, "alpha", dtypes.quint8)
                helper.set_attr_dtype(quantized_conv_node, "Tbias", dtypes.float32)
                                                # if self.device == 'gpu' else dtypes.qint32)
                helper.set_attr_string_list(quantized_conv_node, 'fused_ops', [b'BiasAdd', b'_FusedHardSwish'])
                helper.set_attr_type_list(quantized_conv_node, 'input_types', [
                    input_data_type.as_datatype_enum,
                    dtypes.qint8.as_datatype_enum,
                    dtypes.float32.as_datatype_enum,# if self.device == 'gpu' else dtypes.qint32.as_datatype_enum,
                    dtypes.float32.as_datatype_enum,
                    dtypes.float32.as_datatype_enum,
                    dtypes.float32.as_datatype_enum,
                    dtypes.float32.as_datatype_enum,
                 ])
                helper.set_attr_type_list(quantized_conv_node, 'out_types', [
                                          dtypes.qint32.as_datatype_enum,
                                          dtypes.float32.as_datatype_enum,
                                          dtypes.float32.as_datatype_enum, ])

                self.add_output_graph_node(quantized_conv_node)
                quantize_down_name = self._add_quantize_down_nodes(
                    node, quantized_node_name, dtypes.quint8, is_relu6)
                self._intel_cpu_add_dequantize_result_node(
                    quantize_down_name, match_node_name[5], dtypes.quint8)

            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)

    def apply_newly_conv_biasadd_swishf32_fusion(self, match_node_name):
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
                self.logger.debug("skip node {}".format(node.name))
            elif node.name == match_node_name[0]:
                self.logger.debug("Matched node {} with input {}.".format(node.name, node.input))
                quantized_node_name = node.name + "_eightbit_quantized_depthwise_conv"
                if node.op == "Conv2D":
                    quantized_node_name = node.name + "_eightbit_quantized_conv"

                bias_node_name = self.node_name_mapping[match_node_name[1]].node.input[1]
                swish_node_name = match_node_name[2]
                quantized_node_input_names = all_input_names[:2] + \
                    [bias_node_name] + all_input_names[2:] + control_inputs

                node_op = '_QuantizedDepthwiseConv2D'
                if node.op == 'Conv2D':
                    node_op = "_QuantizedConv2D"
                quantized_conv_node = helper.create_node(node_op, quantized_node_name,
                    quantized_node_input_names)

                helper.copy_attr(quantized_conv_node, "strides", node.attr["strides"])
                helper.copy_attr(quantized_conv_node, "padding", node.attr["padding"])
                if "alpha" in self.node_name_mapping[swish_node_name].node.attr:
                    helper.copy_attr(quantized_conv_node, "alpha",
                    self.node_name_mapping[swish_node_name].node.attr["alpha"])
                if "padding_list" in node.attr:
                    helper.copy_attr(quantized_conv_node, "padding_list",
                    node.attr["padding_list"])
                helper.copy_attr(quantized_conv_node, "dilations", node.attr["dilations"])
                input_data_type = dtypes.quint8 if self._find_relu_node(node) else dtypes.qint8
                helper.set_attr_dtype(quantized_conv_node, "Tinput", input_data_type)
                helper.set_attr_dtype(quantized_conv_node, "Tfilter",dtypes.qint8)
                helper.set_attr_dtype(quantized_conv_node, "Tsummand", dtypes.qint32)
                helper.set_attr_dtype(quantized_conv_node, "out_type", dtypes.qint32)
                helper.set_attr_dtype(quantized_conv_node, "Tbias", dtypes.float32
                                                if self.device == 'gpu' else dtypes.qint32)
                fused_ops = [b'BiasAdd', b'_FusedSwish']
                helper.set_attr_string_list(quantized_conv_node, 'fused_ops', fused_ops)
                helper.set_attr_type_list(quantized_conv_node, 'input_types', [
                    input_data_type.as_datatype_enum,
                    dtypes.qint8.as_datatype_enum,
                    dtypes.float32.as_datatype_enum if self.device == 'gpu' else dtypes.qint32.as_datatype_enum,
                    dtypes.float32.as_datatype_enum,
                    dtypes.float32.as_datatype_enum,
                    dtypes.float32.as_datatype_enum,
                    dtypes.float32.as_datatype_enum,
                 ])
                helper.set_attr_type_list(quantized_conv_node, 'out_types', [
                                          dtypes.qint32.as_datatype_enum,
                                          dtypes.float32.as_datatype_enum,
                                          dtypes.float32.as_datatype_enum, ])
                self.add_output_graph_node(quantized_conv_node)
                quantize_down_name = self._add_quantize_down_nodes(
                    node, quantized_node_name, dtypes.qint8, False)
                self._intel_cpu_add_dequantize_result_node(
                    quantize_down_name, swish_node_name, dtype=dtypes.qint8)


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
                self.logger.info("Unknown fusion pattern {}.".format(fusion_name))
                if self.remove_redundant_quant_flag:
                    self.input_graph = self.remove_redundant_quantization(self.input_graph)
                return self.input_graph, []

            self.input_graph = self.output_graph
            self._reset_output_node_maps()
            if self.remove_redundant_quant_flag:
                self.output_graph = self.remove_redundant_quantization(self.output_graph)

            return self.output_graph, matched_node_name

        return self.input_graph, []
