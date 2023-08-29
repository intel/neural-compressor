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
"""Switch Graph Rewriter."""


from tensorflow.python.framework import tensor_util

from neural_compressor.adaptor.tf_utils.graph_util import GraphAnalyzer
from neural_compressor.utils.utility import dump_elapsed_time

from ..graph_base import GraphRewriterBase


class SwitchOptimizer(GraphRewriterBase):
    """Remove switch op if the input condition is true."""

    @dump_elapsed_time("Pass SwitchOptimizer")
    def do_transformation(self):
        """Replace all enter ops whose output is matmul with const.

        Args:
          input_graph_def (graphdef): graphdef object

        Returns:
           [graphdef]: optimized graph
        """
        cur_graph = GraphAnalyzer()
        cur_graph.graph = self.model

        graph_info = cur_graph.parse_graph()
        target_nodes = cur_graph.query_fusion_pattern_nodes([["Switch"]])

        for node_combination in target_nodes:
            switch_node = graph_info[node_combination[0]].node
            pred_node = graph_info[switch_node.input[1]].node
            if (
                pred_node.op == "Const"
                and tensor_util.MakeNdarray(graph_info[pred_node.name].node.attr["value"].tensor)
            ) or (
                pred_node.op == "PlaceholderWithDefault"
                and tensor_util.MakeNdarray(graph_info[pred_node.input[0]].node.attr["value"].tensor)
            ):
                condition = []
                for output in graph_info[node_combination[0]].outputs:
                    successor_node = graph_info[output].node
                    for index, value in enumerate(successor_node.input):
                        if value == node_combination[0] + ":1":
                            condition.append(True)
                        elif value == node_combination[0] + ":0":
                            condition.append(False)

                if not all(condition):
                    continue

                for output in graph_info[node_combination[0]].outputs:
                    successor_node = graph_info[output].node
                    replace_index = None
                    for index, value in enumerate(successor_node.input):
                        if value == node_combination[0] + ":1":
                            replace_index = index
                            break
                    if not replace_index:
                        break
                    successor_node.input[replace_index] = switch_node.input[0]
                    switch_node_outputs = list(graph_info[node_combination[0]].outputs)
                    if switch_node_outputs.index(output) == len(switch_node_outputs) - 1:
                        cur_graph.remove_node_with_single_input_output(node_combination[0])
            else:
                continue

        return cur_graph.dump_graph()
