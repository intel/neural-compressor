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


from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import dtypes
from neural_compressor.utils.utility import dump_elapsed_time

from ..graph_base import GraphRewriterBase
from ..graph_util import GraphAnalyzer
from ..graph_util import GraphRewriterHelper as Helper


class FuseTransposeReshapeOptimizer(GraphRewriterBase):
    @dump_elapsed_time("Pass FuseTransposeReshapeOptimizer")
    def do_transformation(self):
        g = GraphAnalyzer()
        g.graph = self.model
        graph_info = g.parse_graph()

        patterns = [['Transpose'], ['Reshape'], ['MatMul', 'DepthwiseConv2dNative', 'Conv2D']]

        matched_nodes = g.query_fusion_pattern_nodes(patterns)

        valid_match = []
        for i in matched_nodes:
            transpose_input_node_name = graph_info[i[0]].node.input[0]
            transpose_input_node = graph_info[transpose_input_node_name].node
            if transpose_input_node.op == 'Const':
                valid_match.append(i)

        for i in valid_match:
            transpose_node = graph_info[i[0]].node
            transpose_input_node = graph_info[transpose_node.input[0]].node
            transpose_input_perm = graph_info[transpose_node.input[1]].node
            reshape_node = graph_info[i[1]].node
            reshape_shape_node = graph_info[reshape_node.input[1]].node
            transpose_input_node_content = tensor_util.MakeNdarray(
                transpose_input_node.attr['value'].tensor)
            transpose_perm_node_content = tensor_util.MakeNdarray(
                transpose_input_perm.attr['value'].tensor)
            reshape_shape_node_content = tensor_util.MakeNdarray(
                reshape_shape_node.attr['value'].tensor)

            converted_node = transpose_input_node_content.transpose(
                transpose_perm_node_content).reshape(reshape_shape_node_content)
            g.remove_node(i[0])
            g.remove_node(transpose_input_node.name)
            g.remove_node(transpose_input_perm.name)
            new_node_name = transpose_input_node.name + '_converted'
            new_node = Helper.create_constant_node(
                new_node_name, converted_node, dtype=dtypes.float32, shape=converted_node.shape)
            g.replace_const_node(new_node, [i[2]], i[1])
            g.remove_node(i[1])
            g.remove_node(reshape_shape_node.name)

        return g.dump_graph()
