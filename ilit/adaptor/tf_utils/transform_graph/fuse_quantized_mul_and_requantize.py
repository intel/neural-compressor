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


class FuseQuantizedMulAndRequantize(GraphTransformBase):
    """
    Fuse QuantizedMatMulWithBiasAndRelu with requantize .
    """

    def __init__(self, input_pb):
        super(FuseQuantizedMulAndRequantize, self).__init__(input_pb)

        self.output_name_index_mapping = {}
        self.input_rename = {}

    def get_fuse_index(self, input_node_map, input_name_list):
        matmul_op_list = ["QuantizedMatMulWithBiasAndRelu"]

        fuse_op_list = []

        for node_index, node_name in enumerate(input_name_list):
            node_op = input_node_map[node_name].op
            if node_op in matmul_op_list and \
                    input_node_map[input_name_list[node_index + 1]].op == "Const" and \
                    input_node_map[input_name_list[node_index + 2]].op == "Const" and \
                    input_node_map[input_name_list[node_index + 3]].op == "Requantize":
                fuse_op_list.append(node_index)
        return fuse_op_list

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

    def check_node_existence(self, graph, node_name):
        for node in graph.node:
            if node.name == node_name:
                return node
        return None

    def generate_output_graph(self, input_graph_def, input_node_map,
                              fuse_op_list):
        output_graph_def = graph_pb2.GraphDef()
        skip_list = []
        skip_node_name = []
        uint8_type = dtypes.quint8.as_datatype_enum
        qint32_type = dtypes.qint32.as_datatype_enum
        for index, node in enumerate(input_graph_def.node):

            if index in fuse_op_list:
                input_node = input_node_map[node.input[0]]
                if input_node.op == 'QuantizeV2':
                    new_node = node_def_pb2.NodeDef()

                    new_node.op = node.op + "AndRequantize"
                    for _, value in enumerate(node.input):
                        new_node.input.append(value)
                    weights_node_name = node.input[1]
                    bias_node_name = node.input[2]
                    min_input_node = input_node_map[
                        self.get_node_name_from_input(input_node.input[1])]
                    max_input_node = input_node_map[
                        self.get_node_name_from_input(input_node.input[2])]
                    requantize_node = input_graph_def.node[index + 3]
                    frozen_max_node = input_graph_def.node[index + 2]
                    frozen_min_node = input_graph_def.node[index + 1]
                    new_node.name = requantize_node.name
                    min_filter_node_name = node.input[5]
                    max_filter_node_name = node.input[6]

                    new_node.input.append(frozen_min_node.name)
                    new_node.input.append(frozen_max_node.name)
                    min_filter_node = input_node_map[min_filter_node_name]
                    max_filter_node = input_node_map[max_filter_node_name]

                    new_node.attr["T1"].CopyFrom(node.attr['T1'])
                    new_node.attr["T2"].CopyFrom(node.attr['T2'])
                    min_input_value = (tensor_util.MakeNdarray(
                        min_input_node.attr['value'].tensor))
                    max_input_value = (tensor_util.MakeNdarray(
                        max_input_node.attr['value'].tensor))
                    min_filter_value = (tensor_util.MakeNdarray(
                        min_filter_node.attr['value'].tensor))
                    max_filter_value = (tensor_util.MakeNdarray(
                        max_filter_node.attr['value'].tensor))

                    weights_tensor = tensor_util.MakeNdarray(
                        input_node_map[weights_node_name].attr['value'].tensor)
                    bias_tensor = tensor_util.MakeNdarray(
                        input_node_map[bias_node_name].attr['value'].tensor)
                    bias_scale = 255.0 * 127.0 / (
                        max(abs(max_input_value), abs(min_input_value)) *
                        max(abs(max_filter_value), abs(min_filter_value)))
                    QaAmin = 255 * min_input_value / (max_input_value -
                                                      min_input_value)

                    int32_bias = []

                    for bias_index, value in enumerate(
                            np.sum(np.array(weights_tensor, dtype=np.int32),
                                   axis=0,
                                   dtype=np.int32)):
                        int32_bias.append(
                            int(bias_tensor[bias_index] * bias_scale +
                                value * QaAmin))

                    bias_node = self.check_node_existence(
                        output_graph_def, bias_node_name)
                    if not bias_node:
                        bias_node = input_node_map[bias_node_name]
                    bias_node.attr['dtype'].CopyFrom(
                        attr_value_pb2.AttrValue(type=qint32_type))
                    bias_node.attr['value'].CopyFrom(
                        attr_value_pb2.AttrValue(
                            tensor=tensor_util.make_tensor_proto(
                                int32_bias, dtypes.int32, bias_tensor.shape)))
                    bias_node.attr['value'].tensor.dtype = qint32_type

                    new_node.attr["Tbias"].CopyFrom(
                        attr_value_pb2.AttrValue(type=qint32_type))
                    new_node.attr["Toutput"].CopyFrom(
                        attr_value_pb2.AttrValue(type=uint8_type))

                    skip_list.append(index + 1)
                    skip_list.append(index + 2)
                    skip_list.append(index + 3)
                    output_graph_def.node.extend(
                        [new_node, frozen_max_node, frozen_min_node])
                elif input_node.op == "Requantize":
                    new_node = node_def_pb2.NodeDef()
                    new_node.op = node.op + "AndRequantize"
                    new_node.name = input_graph_def.node[index + 3].name
                    for _, value in enumerate(node.input):
                        new_node.input.append(value)

                    weights_node_name = node.input[1]
                    bias_node_name = node.input[2]
                    min_input_node = input_node_map[
                        self.get_node_name_from_input(input_node.input[3])]
                    max_input_node = input_node_map[
                        self.get_node_name_from_input(input_node.input[4])]
                    requantize_node = input_graph_def.node[index + 3]
                    frozen_max_node = input_graph_def.node[index + 2]
                    frozen_min_node = input_graph_def.node[index + 1]
                    skip_list.append(index + 1)
                    skip_list.append(index + 2)
                    skip_list.append(index + 3)
                    new_node.input.append(frozen_min_node.name)
                    new_node.input.append(frozen_max_node.name)
                    min_filter_node_name = node.input[5]
                    max_filter_node_name = node.input[6]
                    min_filter_node = input_node_map[min_filter_node_name]
                    max_filter_node = input_node_map[max_filter_node_name]

                    new_node.attr["T1"].CopyFrom(node.attr['T1'])
                    new_node.attr["T2"].CopyFrom(node.attr['T2'])
                    min_input_value = (tensor_util.MakeNdarray(
                        min_input_node.attr['value'].tensor))
                    max_input_value = (tensor_util.MakeNdarray(
                        max_input_node.attr['value'].tensor))
                    min_filter_value = (tensor_util.MakeNdarray(
                        min_filter_node.attr['value'].tensor))
                    max_filter_value = (tensor_util.MakeNdarray(
                        max_filter_node.attr['value'].tensor))

                    bias_tensor = tensor_util.MakeNdarray(
                        input_node_map[new_node.input[2]].attr['value'].tensor)
                    bias_scale = 255.0 * 127.0 / (
                        max(abs(max_input_value), abs(min_input_value)) *
                        max(abs(max_filter_value), abs(min_filter_value)))
                    bias_int32 = [int(i * bias_scale) for i in bias_tensor]
                    bias_node = self.check_node_existence(
                        output_graph_def, bias_node_name)
                    if not bias_node:
                        bias_node = input_node_map[bias_node_name]
                    bias_node.attr['dtype'].CopyFrom(
                        attr_value_pb2.AttrValue(type=qint32_type))
                    bias_node.attr['value'].CopyFrom(
                        attr_value_pb2.AttrValue(
                            tensor=tensor_util.make_tensor_proto(
                                bias_int32, dtypes.int32, bias_tensor.shape)))
                    bias_node.attr['value'].tensor.dtype = qint32_type
                    new_node.attr["Tbias"].CopyFrom(
                        attr_value_pb2.AttrValue(type=qint32_type))
                    new_node.attr["Toutput"].CopyFrom(
                        attr_value_pb2.AttrValue(type=uint8_type))

                    output_graph_def.node.extend(
                        [new_node, frozen_max_node, frozen_min_node])
                else:
                    new_node = node_def_pb2.NodeDef()
                    new_node.CopyFrom(node)
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
        Execute the QuantizedMatMulWithBiasAndRelu And Requantize fusion transformation.
        :return: Transformed graph
        """
        input_node_map, _, node_name_list = self.parse_input_graph(
            self.input_graph)
        fuse_op_list = self.get_fuse_index(input_node_map, node_name_list)
        return self.generate_output_graph(self.input_graph, input_node_map,
                                          fuse_op_list)
