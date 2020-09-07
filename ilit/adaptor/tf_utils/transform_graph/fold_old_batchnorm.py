#
#  -*- coding: utf-8 -*-
#
#  Copyright (c) 2019 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import tensor_util

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import dtypes
from .graph_transform_base import GraphTransformBase

import numpy as np


class FuseOldBN(GraphTransformBase):
    def __init__(self, input_pb):
        super(FuseOldBN, self).__init__(input_pb)

    def get_fuse_index(self, input_node_map, input_name_list):
        fuseable_op_list = ['Conv2D', 'DepthwiseConv2dNative']

        fuse_op_name = {}
        for node_index, node_name in enumerate(input_name_list):
            node_op = input_node_map[node_name].op
            if node_op in (
                    "FusedBatchNorm", "BatchNormWithGlobalNormalization"
            ) and input_node_map[
                    input_node_map[node_name].input[0]].op in fuseable_op_list:
                fuse_op_name[input_node_map[node_name].input[0]] = node_name
        return fuse_op_name

    def parse_input_graph(self, input_graph_def):
        node_type_list = []
        node_name_list = []
        input_node_map = {}
        for node in input_graph_def.node:
            node_name_list.append(node.name)
            node_type_list.append(node.op)
            each_node_input = []
            if node.input:
                for _, sub_input in enumerate(node.input):
                    each_node_input.append(sub_input)

            if node.name not in input_node_map:
                input_node_map[node.name] = node
            else:
                self.logger.info('Duplicate node name {}'.format(node.name))

        return input_node_map, node_type_list, node_name_list

    def get_scale_and_offset_values(self, input_node_map, bn_node):
        is_fused = True if bn_node.op == "FusedBatchNorm" else False
        mean_idx = 3 if is_fused else 1
        var_idx = 4 if is_fused else 2
        beta_idx = 2 if is_fused else 3
        gamma_idx = 1 if is_fused else 4
        epsilon_attr = "epsilon" if is_fused else "variance_epsilon"
        # TODO need to check scale_after_normalization
        scale_after_normalization = is_fused

        mean_node = input_node_map[bn_node.input[mean_idx][:]]
        variance_node = input_node_map[bn_node.input[var_idx][:]]
        beta_node = input_node_map[bn_node.input[beta_idx][:]]
        gamma_node = input_node_map[bn_node.input[gamma_idx][:]]

        mean_node_tensor = mean_node.attr['value'].tensor
        variance_node_tensor = variance_node.attr['value'].tensor
        beta_node_tensor = beta_node.attr['value'].tensor
        gamma_node_tensor = gamma_node.attr['value'].tensor
        variance_epsilon = bn_node.attr[epsilon_attr].f

        num_cols = mean_node_tensor.tensor_shape.dim[0].size

        offsets = []
        scales = []
        for i in range(num_cols):
            scale_value = 1.0 / np.sqrt(
                tensor_util.MakeNdarray(variance_node_tensor).flat[i] +
                variance_epsilon)
            if scale_after_normalization:
                scale_value *= tensor_util.MakeNdarray(
                    gamma_node_tensor).flat[i]
            scales.append(scale_value)
            offset_value = tensor_util.MakeNdarray(
                beta_node_tensor).flat[i] - tensor_util.MakeNdarray(
                    mean_node_tensor).flat[i] * scale_value
            offsets.append(offset_value)

        return scales, offsets

    def generate_output_graph(self, input_graph_def, input_node_map,
                              fuse_op_name):
        output_graph_def = graph_pb2.GraphDef()
        skip_list = []
        skip_node_name = []

        for index, node in enumerate(input_graph_def.node):
            if node.name in fuse_op_name:
                conv_node = input_node_map[node.name]
                bn_node = input_node_map[fuse_op_name[node.name]]
                scales, offsets = self.get_scale_and_offset_values(
                    input_node_map, bn_node)
                weights_node_name = conv_node.input[1]
                weights_node = input_node_map[weights_node_name]

                for bn_input in bn_node.input:
                    skip_node_name.append(bn_input)
                skip_node_name.append(bn_node.name)
                new_node = node_def_pb2.NodeDef()
                new_node.op = conv_node.op
                new_node.name = conv_node.name

                for _, value in enumerate(node.input):
                    new_node.input.append(value)
                weights_node_tensor_shape = weights_node.attr[
                    'value'].tensor.tensor_shape
                if conv_node.op == 'Conv2D':
                    weights_cols = weights_node_tensor_shape.dim[3].size
                elif conv_node.op == "DepthwiseConv2dNative":
                    weights_cols = weights_node_tensor_shape.dim[
                        2].size * weights_node_tensor_shape.dim[3].size
                else:
                    weights_cols = weights_node_tensor_shape.dim[1].size

                weights_tensor = tensor_util.MakeNdarray(
                    weights_node.attr['value'].tensor)

                new_weights = []
                for index, i in enumerate(weights_tensor.flat):
                    new_weights_value = weights_tensor.flat[index] * scales[
                        index % weights_cols]
                    new_weights.append(new_weights_value)

                new_bn = []
                for index in range(weights_cols):
                    new_bn_value = offsets[index]
                    new_bn.append(new_bn_value)

                weights_node.attr['value'].CopyFrom(
                    attr_value_pb2.
                    AttrValue(tensor=tensor_util.make_tensor_proto(
                        new_weights, dtypes.float32, weights_tensor.shape)))

                bias_offset_node = node_def_pb2.NodeDef()
                bias_offset_node.op = "Const"
                bias_offset_node.name = conv_node.name + "_bn_offset"
                bias_offset_node.attr["dtype"].CopyFrom(
                    attr_value_pb2.AttrValue(
                        type=dtypes.float32.as_datatype_enum))
                bias_offset_node.attr["value"].CopyFrom(
                    attr_value_pb2.AttrValue(
                        tensor=tensor_util.make_tensor_proto(
                            new_bn, dtypes.float32, [weights_cols])))

                biasadd_node = node_def_pb2.NodeDef()
                biasadd_node.op = "BiasAdd"
                biasadd_node.name = bn_node.name

                if "data_format" in conv_node.attr:
                    biasadd_node.attr["data_format"].CopyFrom(
                        conv_node.attr['data_format'])
                biasadd_node.attr["T"].CopyFrom(conv_node.attr['T'])
                biasadd_node.input.append(conv_node.name)
                biasadd_node.input.append(bias_offset_node.name)

                for key in conv_node.attr:
                    new_node.attr[key].CopyFrom(conv_node.attr[key])

                output_graph_def.node.extend(
                    [weights_node, bias_offset_node, biasadd_node, new_node])
            elif index in skip_list or node.name in skip_node_name:
                continue
            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                output_graph_def.node.extend([new_node])

        return output_graph_def

    def do_transformation(self):
        """
        Execute the Conv2D/DepthwiseConv2dNative + FusedBatchNorm/BatchNormWithGlobalNormalization
        fusion.
        :return: Transformed graph
        """
        input_node_map, _, node_name_list = self.parse_input_graph(
            self.input_graph)
        fuse_op_name = self.get_fuse_index(input_node_map, node_name_list)
        return self.generate_output_graph(self.input_graph, input_node_map,
                                          fuse_op_name)
