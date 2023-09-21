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
"""Fetch Weight from Reshape Graph Rewriter."""


import numpy as np
from tensorflow.python.framework import dtypes

from neural_compressor.adaptor.tf_utils.graph_util import GraphAnalyzer
from neural_compressor.adaptor.tf_utils.graph_util import GraphRewriterHelper as Helper
from neural_compressor.utils.utility import dump_elapsed_time

from ..graph_base import GraphRewriterBase


class FetchWeightFromReshapeOptimizer(GraphRewriterBase):
    """Handle the Pack + Reshape + Conv2D fusion pattern."""

    @dump_elapsed_time("Pass FetchWeightFromReshapeOptimizer")
    def do_transformation(self):
        """Fetch weight of Conv2D from Pack+Reshape+Conv2D pattern.

        Args:
          input_graph_def (graphdef): graphdef object
        Returns:
           [graphdef]: optimized graph
        """
        cur_graph = GraphAnalyzer()
        cur_graph.graph = self.model

        graph_info = cur_graph.parse_graph()
        target_nodes = cur_graph.query_fusion_pattern_nodes([["Pack"], ["Reshape"], ["Conv2D"]])

        for i, node_combination in enumerate(target_nodes):
            pack_node = graph_info[node_combination[0]].node
            reshape_node = graph_info[node_combination[1]].node
            shape_node = graph_info[reshape_node.input[1]].node
            conv_node = graph_info[node_combination[2]].node
            if not (pack_node.op == "Pack" and reshape_node.op == "Reshape" and conv_node.op == "Conv2D"):
                continue
            reshape_outputs_length = len(graph_info[node_combination[1]].outputs)
            unpack_values = []
            for index in range(pack_node.attr["N"].i):
                values_node = graph_info[pack_node.input[index]].node
                if values_node.op == "Const":
                    unpack_values.append(Helper.values_from_const(values_node))
            input_reshape = np.stack(unpack_values, axis=pack_node.attr["axis"].i)
            if shape_node.op != "Const":
                continue
            shape = Helper.values_from_const(shape_node)
            weight = np.reshape(input_reshape, shape)
            weight_node = Helper.create_constant_node(
                reshape_node.name + "/weight" + "_" + str(i), weight, dtypes.float32
            )
            if i > 0:
                conv_node_j = graph_info[target_nodes[i - 1][2]].node
                graph_info[node_combination[1]].outputs.remove(conv_node_j.name)
            for output in graph_info[node_combination[1]].outputs:
                successor_node = graph_info[output].node
                replace_index = None
                for index, value in enumerate(successor_node.input):
                    if value == reshape_node.name or value == reshape_node.name + "/weight" + "_" + str(i - 1):
                        replace_index = index
                        break
                # weight->conv2d
                cur_graph.add_node(weight_node, None, [successor_node.name])
                successor_node.input[replace_index] = weight_node.name

            if i + 1 == reshape_outputs_length:
                cur_graph.remove_node(reshape_node.name)
                cur_graph.remove_node(values_node.name)
                cur_graph.remove_node(shape_node.name)
                cur_graph.remove_node(pack_node.name)

        return cur_graph.dump_graph()
