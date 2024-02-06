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
"""ExpandDims Graph Rewriter."""


import numpy as np
from tensorflow.python.framework import dtypes

from neural_compressor.tensorflow.quantization.utils.graph_util import GraphAnalyzer
from neural_compressor.tensorflow.quantization.utils.graph_util import GraphRewriterHelper as Helper
from neural_compressor.tensorflow.utils import dump_elapsed_time

from ..graph_base import GraphRewriterBase


class ExpandDimsOptimizer(GraphRewriterBase):
    """Calculate ExpandDims and remove it if its input is weight and next node is Conv2D."""

    @dump_elapsed_time("Pass ExpandDimsOptimizer")
    def do_transformation(self):
        """Handle all ExpandDims ops whose input is weight and output is Conv2D.

        Args:
          input_graph_def (graphdef): graphdef object

        Returns:
           [graphdef]: optimized graph
        """
        cur_graph = GraphAnalyzer()
        cur_graph.graph = self.model

        graph_info = cur_graph.parse_graph()
        target_nodes = cur_graph.query_fusion_pattern_nodes([["ExpandDims"]])

        for node_combination in target_nodes:
            expanddims_node = graph_info[node_combination[0]].node
            dims_node = graph_info[expanddims_node.input[1]].node
            next_node = graph_info[graph_info[node_combination[0]].outputs[0]].node
            # to solve the case that input 0 of ExpandDims is a tensor, not a node
            if expanddims_node.input[0] in graph_info:
                weight_node = graph_info[expanddims_node.input[0]].node
            else:
                continue

            if weight_node.op == "Const" and next_node.op == "Conv2D":
                dims = Helper.values_from_const(dims_node)
                weight_value = np.array(Helper.values_from_const(weight_node))
                new_weight_value = np.expand_dims(weight_value, axis=dims)
                cur_graph.remove_node(weight_node.name)
                new_weight_node = Helper.create_constant_node(weight_node.name, new_weight_value, dtypes.float32)

                for output in graph_info[node_combination[0]].outputs:
                    successor_node = graph_info[output].node
                    replace_index = None
                    for index, value in enumerate(successor_node.input):
                        if value == expanddims_node.name:
                            replace_index = index
                            break
                    # weight->conv2d
                    cur_graph.add_node(new_weight_node, None, [successor_node.name])
                    successor_node.input[replace_index] = new_weight_node.name
                # remove ExpandDims and weight_node
                cur_graph.remove_node(expanddims_node.name)
            else:
                continue

        return cur_graph.dump_graph()
