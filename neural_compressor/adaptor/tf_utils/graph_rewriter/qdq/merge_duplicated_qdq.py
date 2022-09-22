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


from tensorflow.core.framework import node_def_pb2
from neural_compressor.utils.utility import dump_elapsed_time

from ..graph_base import GraphRewriterBase
from neural_compressor.adaptor.tf_utils.graph_util import GraphAnalyzer
from neural_compressor.adaptor.tf_utils.graph_util import GraphRewriterHelper as Helper


class MergeDuplicatedQDQOptimizer(GraphRewriterBase):

    @dump_elapsed_time("Pass MergeDuplicatedQDQOptimizer")
    def do_transformation(self):
        cur_graph = GraphAnalyzer()
        cur_graph.graph = self.model
        graph_info = cur_graph.parse_graph()

        patterns = [['QuantizeV2'], ['Dequantize']]
        matched_nodes = cur_graph.query_fusion_pattern_nodes(patterns)

        quantizev2_input_map = {}
        dequantize_map = {}
        for i in matched_nodes:
            quantizev2_input_node_name = graph_info[i[0]].node.input[0]
            if quantizev2_input_node_name in quantizev2_input_map:
                quantizev2_input_map[quantizev2_input_node_name].append(graph_info[i[0]].node)
                dequantize_map[quantizev2_input_node_name].append(graph_info[i[1]].node)
            else:
                quantizev2_input_map[quantizev2_input_node_name] = [graph_info[i[0]].node]
                dequantize_map[quantizev2_input_node_name] = [graph_info[i[1]].node]

        for input_map_node_name in quantizev2_input_map.keys():
            if input_map_node_name not in cur_graph.node_name_details:
                continue

            # the nodes share the the same input and have the same QuantizeV2 op type
            if len(quantizev2_input_map[input_map_node_name]) > 1:
                # merge the duplicated QuantizeV2 op and Dequantize op.
                new_quantize_node = quantizev2_input_map[input_map_node_name][0]
                new_dequantize_node = dequantize_map[input_map_node_name][0]

                # set the new QuantizeV2 node as the only output of the parent node
                for i in quantizev2_input_map[input_map_node_name]:
                    if i.name != new_quantize_node.name:
                        cur_graph.node_name_details[input_map_node_name].outputs.remove(i.name)

                # set the new Dequantized node as the input of the end nodes
                for i in dequantize_map[input_map_node_name]:
                    if i.name != new_dequantize_node.name:
                        end_node_name = cur_graph.node_name_details[i.name].outputs[0]
                        cur_graph.node_name_details[end_node_name].node.input.remove(i.name)
                        cur_graph.node_name_details[end_node_name].node.input.insert(0, new_dequantize_node.name)
                        cur_graph.node_name_details[new_dequantize_node.name].outputs.append(end_node_name)

                # remove the duplicated quantized nodes
                for i in quantizev2_input_map[input_map_node_name]:
                    if i.name != new_quantize_node.name:
                        # remove quantize min node
                        cur_graph.remove_node(cur_graph.node_name_details[i.name].node.input[1])
                        # remove quantize max node
                        cur_graph.remove_node(cur_graph.node_name_details[i.name].node.input[2])
                        # remove quantize node
                        cur_graph.remove_node(i.name)

                # remove the duplicated dequantized nodes
                for i in dequantize_map[input_map_node_name]:
                    if i.name != new_dequantize_node.name:
                        # remove dequantize node
                        cur_graph.remove_node(i.name)

        return cur_graph.dump_graph()
