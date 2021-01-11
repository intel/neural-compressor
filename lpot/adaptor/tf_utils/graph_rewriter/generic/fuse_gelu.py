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

from ..graph_base import GraphRewriterBase
from ..graph_util import GraphAnalyzer
from ..graph_util import GraphRewriterHelper as Helper


class FuseGeluOptimizer(GraphRewriterBase):
    """Fuse Sqrt + RealDiv + Erf + AddV2 + Mul + Mul into Gelu op.
    """

    def do_transformation(self):
        cur_graph = GraphAnalyzer()
        cur_graph.graph = self.model

        graph_info = cur_graph.parse_graph()

        target_nodes = cur_graph.query_fusion_pattern_nodes(
            [["Sqrt"], ["RealDiv"], ["Erf"], ["AddV2"], ["Mul"], ["Mul"]])

        for node_combination in target_nodes:

            sqrt_node = graph_info[node_combination[0]].node
            realdiv_node = graph_info[node_combination[1]].node
            erf_node = graph_info[node_combination[2]].node
            addv2_node = graph_info[node_combination[3]].node
            mul1_node = graph_info[node_combination[4]].node
            mul2_node = graph_info[node_combination[5]].node

            sqrt_input_name = Helper.node_name_from_input(sqrt_node.input[0])
            sqrt_value = graph_info[sqrt_input_name].node.attr['value'].tensor.float_val[0]

            if sqrt_value != 2:
                continue
            
            addv2_value = None
            mul1_value = None
            gelu_input_name = None
            gelu_input_name = None
            gelu_input_name = None
            for i in realdiv_node.input:
                i = Helper.node_name_from_input(i)
                if i != sqrt_node.name:
                    gelu_input_name = i
                    break
            
            addv2_const_name = None
            for i in addv2_node.input:
                i = Helper.node_name_from_input(i)
                if i != erf_node.name:
                    addv2_value = graph_info[i].node.attr['value'].tensor.float_val[0]
                    addv2_const_name = i
                    break
            
            if addv2_value != 1:
                continue

            mul1_const_name = None
            for i in mul1_node.input:
                i = Helper.node_name_from_input(i)
                if i != addv2_node.name:
                    mul1_value = graph_info[i].node.attr['value'].tensor.float_val[0]
                    mul1_const_name = i
                    break

            if mul1_value != 0.5:
                continue

            cur_graph.remove_node(sqrt_node.input[0])
            cur_graph.remove_node(sqrt_node.name)
            cur_graph.remove_node(realdiv_node.name)
            cur_graph.remove_node(erf_node.name)
            cur_graph.remove_node(addv2_node.name)
            cur_graph.remove_node(mul1_node.name)
            cur_graph.remove_node(addv2_const_name)
            cur_graph.remove_node(sqrt_input_name)
            cur_graph.remove_node(mul1_const_name)

            gelu_node = Helper.create_node("Gelu", sqrt_node.name, [gelu_input_name])

            cur_graph.add_node(gelu_node, gelu_input_name, graph_info[mul2_node.name].outputs)
            cur_graph.remove_node(mul2_node.name)

        return cur_graph.dump_graph()
