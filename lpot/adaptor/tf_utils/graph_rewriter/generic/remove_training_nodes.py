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

from lpot.utils.utility import dump_elapsed_time

from ..graph_base import GraphRewriterBase
from ..graph_util import GraphAnalyzer


class RemoveTrainingNodesOptimizer(GraphRewriterBase):
    def __init__(self, model, protected_nodes=[], types_to_splice=['Identity', 'CheckNumerics']):
        super().__init__(model)
        self.protected_nodes = protected_nodes
        self.types_to_splice = types_to_splice

    @dump_elapsed_time("Pass RemoveTrainingNodesOptimizer")
    def do_transformation(self):
        graph_handle = GraphAnalyzer()
        graph_handle.graph = self.model

        graph_info = graph_handle.parse_graph()
        # input_nodes = input_graph.node

        control_input_names = set()
        node_names_with_control_input = set()
        names_to_splice = {}

        for node_name, v in graph_info.items():
            for node_input in v.node.input:
                if "^" in node_input:
                    control_input_names.add(node_input.replace("^", ""))
                    node_names_with_control_input.add(node_name)

        for node_name, v in graph_info.items():
            if v.node.op in self.types_to_splice and v.node.name not in self.protected_nodes:
                # We don't want to remove nodes that have control edge inputs, because
                # they might be involved in subtle dependency issues that removing them
                # will jeopardize.
                if node_name not in node_names_with_control_input:
                    names_to_splice[node_name] = v.node.input[0]

        # We also don't want to remove nodes which are used as control edge inputs.
        names_to_splice = {
            name: value
            for name, value in names_to_splice.items()
            if name not in control_input_names
        }
        for k, _ in names_to_splice.items():
            graph_handle.remove_node_with_single_input_output(k)

        return graph_handle.dump_graph()
