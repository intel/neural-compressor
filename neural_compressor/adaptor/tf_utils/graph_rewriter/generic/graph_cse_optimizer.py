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
"""CSE Graph Rewriter."""

from tensorflow.core.framework import graph_pb2

from neural_compressor.adaptor.tf_utils.graph_util import GraphAnalyzer
from neural_compressor.adaptor.tf_utils.graph_util import GraphRewriterHelper as Helper
from neural_compressor.utils.utility import dump_elapsed_time

from ..graph_base import GraphRewriterBase


class GraphCseOptimizer(GraphRewriterBase):
    """We introduce the CSE optimizer to optimize the nodes that contains identical op type.

    Case 1. Node A has three output nodes(B,C,D) and those child nodes has their own outputs
            (B1/C1C2/D1).
            Node A
            x    x    x
            x    x     x
    NODE B   NODE C  NODE D
    x        x   x     x
    B1       C1  C2    D1
    If Node B/C/D have the identical memory-bound op, like relu/relu6. The graph will be
    converted as below.
    We removed the Node C & Node D, updated the B as the input of C1/C2/D1.
            Node A
            x
            Node B
            x x  x  x
            x  x   x  x
        x   x    x  x
        B1  C1   C2  D1
    Case 2.  Node A has three output nodes(B,C,D) and those child nodes has their own outputs
            (B1/C1C2/D1).
            Node A
            x   x    x
            x    x     x
    NODE B   NODE C  NODE D
        x      x   x     x
        B1     C1  C2    D1
    If Node B and C have the identical memory-bound op, like relu/relu6. The graph will be
    converted as below.
    We removed the Node C, updated the B as the input of C1/C2.
                Node A
                x     x
            Node B  Node D
            x   x  x    x
            x   |   x    x
            x   |    x    x
            B1   C1    C2   D1

    Returns:
        [graphdef]: A optimized graphdef object.
    """

    computational_op_type = ("Conv2D", "Conv3D", "DepthwiseConv2dNative", "MatMul")

    @dump_elapsed_time("Pass GraphCseOptimizer")
    def do_transformation(self):
        """Optimize the graph contains multi output nodes.

        If those nodes' type are identical, those nodes should be elimated.
        Currently, we supported memory bound ops only.

        Args:
            input_graph_def (graphdef): graphdef object

        Returns:
            [graphdef]: optimized graph
        """
        GraphAnalyzer().graph = self.model

        graph_info = GraphAnalyzer().parse_graph()

        need_to_update_node = []
        # TODO Enhance below code snippet by using recursive method.
        for _, i in graph_info.items():
            candidate_node = [
                graph_info[child_name].node
                for child_name in i.outputs
                if graph_info[child_name].node.op not in self.computational_op_type
            ]
            candidate_node_unique_type = set([i.op for i in candidate_node])
            if len(candidate_node_unique_type) == len(candidate_node):
                # it means each sub node has their own type.
                continue
            node_type_name_mapping = {}
            # Created dict which key is op type and value is node has identical op type.
            for each_node in candidate_node:
                node_type = each_node.op
                node_name = each_node.name
                if node_type not in node_type_name_mapping:
                    node_type_name_mapping[node_type] = [node_name]
                else:
                    node_type_name_mapping[node_type].append(node_name)

            for _, node_names in node_type_name_mapping.items():
                # ignore unique op type and node with multi-outputs
                if len(node_names) == 1 or len(graph_info[node_names[0]].outputs) > 1:
                    continue
                # TODO Need to enhance below algorithm before golden.
                filter_node = [node_names[0]]
                for sub_node_name in node_names[1:]:
                    if not Helper.compare_node_attr(graph_info[node_names[0]].node, graph_info[sub_node_name].node):
                        continue
                    filter_node.append(sub_node_name)

                need_to_update_node.append({i.node.name: filter_node})

        for node_pair in need_to_update_node:
            for upper_node_name, lower_node_name in node_pair.items():
                keep_sub_node_name = lower_node_name[0]
                for removeable_node_name in lower_node_name[1:]:
                    graph_info[upper_node_name].outputs.remove(removeable_node_name)
                    for grand_child_node_name in graph_info[removeable_node_name].outputs:
                        filter_input_name = [
                            Helper.node_name_from_input(i) for i in graph_info[grand_child_node_name].node.input
                        ]
                        replace_index = filter_input_name.index(removeable_node_name)
                        graph_info[grand_child_node_name].node.input[replace_index] = keep_sub_node_name
                        graph_info[grand_child_node_name].node.input[replace_index] = keep_sub_node_name
                    graph_info.pop(removeable_node_name)

        output_graph_def = graph_pb2.GraphDef()

        for _, node_info in graph_info.items():
            output_graph_def.node.extend([node_info.node])

        return output_graph_def
