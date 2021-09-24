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

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from neural_compressor.utils.utility import dump_elapsed_time

from ..graph_base import GraphRewriterBase
from ..graph_util import GraphAnalyzer
from ..graph_util import GraphRewriterHelper as Helper


class ConvertLeakyReluOptimizer(GraphRewriterBase):
    """ Convert below subgraph to Node A + LeakyRelu.
                Node A             Node A
                  |    \              |
                  |     \             |
                  |     Mul  --->     |
                  |      /            |
                  |     /             |
                 Maximum           LeakyRelu
        Note, the coefficient of Mul should be less than 1 or the conversion is not valid.
    """
    @dump_elapsed_time("Pass ConvertLeakyReluOptimizer")
    def do_transformation(self):
        g = GraphAnalyzer()
        g.graph = self.model
        graph_info = g.parse_graph()
        target_nodes = g.query_fusion_pattern_nodes([['Mul'], ['Maximum']])
        for i in target_nodes:
            successor_node_names = graph_info[i[1]].outputs

            mul_input_names = list(graph_info[i[0]].node.input)
            max_input_names = list(graph_info[i[1]].node.input)
            common_input = list(set(mul_input_names).intersection(set(max_input_names)))

            if len(common_input) != 1:
                continue
            mul_coeff_node_name = list(set(mul_input_names).difference(set(max_input_names)))[0]
            mul_coeff_node = graph_info[mul_coeff_node_name].node
            if mul_coeff_node.op != 'Const':
                continue

            alpha_value = float(tensor_util.MakeNdarray(mul_coeff_node.attr['value'].tensor))
            if alpha_value > 1.0:
                continue

            leaky_relu_node_name = i[1] + '_leakyrelu'
            leaky_relu_node = Helper.create_node('LeakyRelu', leaky_relu_node_name, common_input)
            Helper.set_attr_dtype(leaky_relu_node, "T", dtypes.float32)
            Helper.set_attr_float(leaky_relu_node, "alpha", alpha_value)

            g.replace_single_node(leaky_relu_node, common_input, i[1], successor_node_names, i[1])
            g.remove_node(i[1])
            g.remove_node(i[0])
            g.remove_node(mul_coeff_node_name)

        return g.dump_graph()
