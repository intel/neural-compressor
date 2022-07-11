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

from neural_compressor.utils.utility import dump_elapsed_time
from ..graph_base import GraphRewriterBase
from neural_compressor.adaptor.tf_utils.graph_util import GraphAnalyzer

class LiftQDQAboveReshapeTransposeOptimizer(GraphRewriterBase): # pragma: no cover
    @dump_elapsed_time("Pass LiftQDQAboveReshapeTransposeOptimizer")
    def do_transformation(self):
        g = GraphAnalyzer()
        g.graph = self.model
        graph_info = g.parse_graph()

        patterns = [['Reshape'], ['Transpose'], ['QuantizeV2'], ['Dequantize']]
        matched_nodes = g.query_fusion_pattern_nodes(patterns)

        for i in matched_nodes:
            reshape_input_node_name = graph_info[i[0]].node.input[0]
            reshape_node_name = graph_info[i[0]].node.name
            transpose_node_name = graph_info[i[1]].node.name
            quantizev2_node_name = graph_info[i[2]].node.name
            dequantize_node_name = graph_info[i[3]].node.name
            dequantize_output_node_name = g.node_name_details[dequantize_node_name].outputs[0]

            g.node_name_details[reshape_input_node_name].outputs.remove(reshape_node_name)
            g.node_name_details[reshape_input_node_name].outputs.append(quantizev2_node_name)
            g.node_name_details[quantizev2_node_name].node.input[0] = reshape_input_node_name
            g.node_name_details[dequantize_node_name].outputs.remove(dequantize_output_node_name)
            g.node_name_details[dequantize_node_name].outputs.append(reshape_node_name)
            g.node_name_details[reshape_node_name].node.input[0] = dequantize_node_name
            g.node_name_details[transpose_node_name].outputs.remove(quantizev2_node_name)
            g.node_name_details[transpose_node_name].outputs.append(dequantize_output_node_name)
            g.node_name_details[dequantize_output_node_name].node.input.remove(dequantize_node_name)
            g.node_name_details[dequantize_output_node_name].node.input.append(transpose_node_name)

        return g.dump_graph()
