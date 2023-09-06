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
"""Split shared input Graph Rewriter."""

from tensorflow.core.framework import node_def_pb2

from neural_compressor.adaptor.tf_utils.graph_util import GraphAnalyzer
from neural_compressor.adaptor.tf_utils.graph_util import GraphRewriterHelper as Helper
from neural_compressor.utils.utility import dump_elapsed_time

from ..graph_base import GraphRewriterBase


class SplitSharedInputOptimizer(GraphRewriterBase):
    """Split the shared input if the input node is shared and const."""

    @dump_elapsed_time("Pass SplitSharedInputOptimizer")
    def do_transformation(self):
        """Execute splitting the shared input."""
        cur_graph = GraphAnalyzer()
        cur_graph.graph = self.model

        graph_info = cur_graph.parse_graph()

        is_shared_input = False
        # map of: input_name - op_name
        input_map = {}
        for node_name in list(graph_info.keys()):
            node = graph_info[node_name].node
            for _, input_node_name in enumerate(node.input):
                if input_node_name.startswith("^"):
                    continue
                if graph_info[Helper.node_name_from_input(input_node_name)].node.op == "Const":
                    # is shared and current node is not the first one
                    # sharing the input
                    if input_node_name in input_map:
                        is_shared_input = True
                        input_map[input_node_name].append(node.name)
                        new_input_node = node_def_pb2.NodeDef()
                        new_input_node.CopyFrom(graph_info[input_node_name].node)
                        new_input_node.name = input_node_name + "_nc_share_" + str(len(input_map[input_node_name]))
                        cur_graph.replace_const_node(new_input_node, [node.name], input_node_name, False)
                    else:
                        input_map[input_node_name] = [node.name]

        return cur_graph.dump_graph() if is_shared_input else self.model
