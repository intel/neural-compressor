#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
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
"""Share QDQ for ITEX Y pattern Graph Rewriter."""

from neural_compressor.adaptor.tf_utils.graph_util import GraphAnalyzer
from neural_compressor.utils.utility import dump_elapsed_time

from ..graph_base import GraphRewriterBase


class ShareQDQForItexYPatternOptimizer(GraphRewriterBase):
    """Insert Q/DQ op before one input of Add to enable Conv/MatMul + BiasAdd + Add + Relu fusion for ITEX.

    Only 1 Q/DQ before Add op need to be inserted. Insert 2 Q/DQ breaks the ITEX fusion pattern.
    """

    @dump_elapsed_time("Pass ShareQDQForItexYPatternOptimizer")
    def do_transformation(self):
        """Share the QDQ of one output of Relu node with the another output which is Add node."""
        g = GraphAnalyzer()
        g.graph = self.model
        graph_info = g.parse_graph()

        patterns = [["Relu", "MaxPool"], ["QuantizeV2"], ["Dequantize"]]
        matched_nodes = g.query_fusion_pattern_nodes(patterns)

        for i in matched_nodes:
            relu_node_name = graph_info[i[0]].node.name
            if len(g.node_name_details[relu_node_name].outputs) != 2:
                continue

            add_node_name = g.node_name_details[relu_node_name].outputs[0]
            quantize_node_name = g.node_name_details[relu_node_name].outputs[1]
            if (
                "Add" not in g.node_name_details[add_node_name].node.op
                or g.node_name_details[quantize_node_name].node.op != "QuantizeV2"
            ):
                continue
            dequantize_node_name = graph_info[i[2]].node.name

            g.node_name_details[relu_node_name].outputs.remove(add_node_name)
            g.node_name_details[dequantize_node_name].outputs.append(add_node_name)
            g.node_name_details[add_node_name].node.input.remove(relu_node_name)
            g.node_name_details[add_node_name].node.input.append(dequantize_node_name)

        return g.dump_graph()
