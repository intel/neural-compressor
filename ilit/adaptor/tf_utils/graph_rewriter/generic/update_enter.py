#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Intel Corporation
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


class UpdateEnterOptimizer(GraphRewriterBase):
    """ This is a workaround of control ops
        exclude all nodes following Enters
    """
    supported_ops = ["MatMul", "BiasAdd"]

    def __init__(self, model):
        super().__init__(model)
        self.graph_analyzer = GraphAnalyzer()
        self.graph_analyzer.graph = self.model
        self.exclude_node_names = []
        self.graph_info = self.graph_analyzer.parse_graph()

    def do_transformation(self):
        """ replace all enter ops whose output is matmul with const

        Args:
          input_graph_def (graphdef): graphdef object

        Returns:
           [graphdef]: optimized graph
        """
        for _, node_detail in self.graph_info.copy().items():
            if node_detail.outputs:
                if node_detail.node.op == "Enter" and self.graph_info[node_detail.outputs[0]
                                                      ].node.op in self.supported_ops:
                    self.exclude_node_names.append(node_detail.outputs[0])

        output_graph_def = self.graph_analyzer.dump_graph()

        return output_graph_def, self.exclude_node_names
