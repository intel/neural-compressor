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
from ..quantize_graph.quantize_graph_common import QuantizeGraphHelper as helper


class FuseColumnWiseMul(GraphTransformBase):
    def __init__(self, input_pb):
        super(FuseColumnWiseMul, self).__init__(input_pb)

    def get_fuse_index(self, input_node_map, input_name_list):
        fuseable_op_list = ['Conv2D', 'DepthwiseConv2dNative', 'MatMul']

        fuse_op_name = {}
        for input_node in input_name_list:
            node_op = input_node_map[input_node].op
            input_node = helper.node_name_from_input(input_node)

            if node_op == "Mul":
                inputs = [helper.node_name_from_input(input) for
                          input in input_node_map[input_node].input]
                if input_node_map[inputs[0]].op in fuseable_op_list and \
                   input_node_map[inputs[1]].op == "Const":

                    fuse_op_name[input_node_map[input_node].input[0]] = input_node

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

    def generate_output_graph(self, input_graph_def, input_node_map,
                              fuse_op_name):
        output_graph_def = graph_pb2.GraphDef()
        skip_list = []
        skip_node_name = []
        for index, node in enumerate(input_graph_def.node):

            if node.name in fuse_op_name:
                skip_list.append(index + 1)

                original_node = input_node_map[node.name]
                mul_node = input_node_map[fuse_op_name[node.name]]
                weights_node_name = original_node.input[1]
                weights_node = input_node_map[weights_node_name]
                mul_value_node_name = mul_node.input[1]
                mul_value_node = input_node_map[mul_value_node_name]

                new_node = node_def_pb2.NodeDef()
                new_node.op = original_node.op
                new_node.name = mul_node.name

                for _, value in enumerate(node.input):
                    new_node.input.append(value)

                if original_node.op == "DepthwiseConv2dNative":
                    weights_col = weights_node.attr[
                        'value'].tensor.tensor_shape.dim[
                            2].size * weights_node.attr[
                                'value'].tensor.tensor_shape.dim[3].size
                elif original_node.op == "Conv2D":
                    weights_col = weights_node.attr[
                        'value'].tensor.tensor_shape.dim[3].size
                else:
                    weights_col = weights_node.attr[
                        'value'].tensor.tensor_shape.dim[1].size
                mul_value_node_tensor = mul_value_node.attr['value'].tensor
                weights_node_tensor = weights_node.attr['value'].tensor

                if len(mul_value_node_tensor.tensor_shape.dim
                       ) != 1 or mul_value_node_tensor.tensor_shape.dim[
                           0].size != weights_col:
                    self.logger.info("Invalid Mul OP fusion.")

                mul_value_node_list = [
                    i for i in tensor_util.MakeNdarray(
                        mul_value_node_tensor).flat
                ]
                new_weights = []
                for index, i in enumerate(
                        tensor_util.MakeNdarray(weights_node_tensor).flat):
                    new_weights_value = i * mul_value_node_list[
                        index % len(mul_value_node_list)]
                    new_weights.append(new_weights_value)

                weights_node.attr['value'].CopyFrom(
                    attr_value_pb2.
                    AttrValue(tensor=tensor_util.make_tensor_proto(
                        new_weights, dtypes.float32,
                        tensor_util.MakeNdarray(weights_node_tensor).shape)))
                skip_node_name.append(weights_node.name)
                output_graph_def.node.extend([weights_node])
                for key in original_node.attr:
                    new_node.attr[key].CopyFrom(original_node.attr[key])

                output_graph_def.node.extend([new_node])

            elif index in skip_list or node.name in skip_node_name:
                continue
            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                output_graph_def.node.extend([new_node])
        return output_graph_def

    def do_transformation(self):
        """
        Execute the Conv2D/DepthwiseConv2dNative/Matmul + Mul fusion.
        :return: Transformed graph
        """
        input_node_map, _, node_name_list = self.parse_input_graph(
            self.input_graph)
        fuse_op_name = self.get_fuse_index(input_node_map, node_name_list)
        return self.generate_output_graph(self.input_graph, input_node_map,
                                          fuse_op_name)
