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

from tensorflow.python.framework import tensor_util

from ..graph_base import GraphRewriterBase
from ..graph_util import GraphAnalyzer
from ..graph_util import GraphRewriterHelper as Helper


class FusePadWithConv2DOptimizer(GraphRewriterBase):
    """Fuse Pad op into Conv2D
    Pad + Conv2D --> Conv2D
    """

    def __init__(self, model, excluded_op_names, inputs, cfg):
        super().__init__(model)
        self.excluded_conv = excluded_op_names
        self.inputs = inputs
        self.cfg = cfg

    def do_transformation(self):
        cur_graph = GraphAnalyzer()
        cur_graph.graph = self.model

        graph_info = cur_graph.parse_graph()
        # TODO Need to remove below code once the tensorflow enhances the shape inference.
        add_nodes = [i for i in cur_graph.query_fusion_pattern_nodes(
            [['Add', 'AddV2'], ('Relu', 'Relu6')]) if len(i) == 2]
        excluded_pad = []
        for sub_add in add_nodes:
            left_path_pad_names = []
            right_path_pad_names = []

            input_node = graph_info[sub_add[0]].node

            for index, each_sub_path in enumerate(input_node.input):
                input_node = graph_info[Helper.node_name_from_input(each_sub_path)].node
                while True:
                    if not input_node.input:
                        break
                    input_node = graph_info[Helper.node_name_from_input(input_node.input[0])].node
                    if input_node.op == 'Pad':
                        if index == 0:
                            left_path_pad_names.append(input_node.name)
                        else:
                            right_path_pad_names.append(input_node.name)
                    if len(graph_info[Helper.node_name_from_input(input_node.name)].outputs) > 1:
                        break

                    if input_node.name in self.inputs:
                        break

            if len(left_path_pad_names) != len(right_path_pad_names) and \
                len(left_path_pad_names) > 0 and len(right_path_pad_names) > 0:
                excluded_pad.extend(left_path_pad_names)
                excluded_pad.extend(right_path_pad_names)
            elif len(left_path_pad_names) == 1 and len(right_path_pad_names) == 1:
                l_padding_tensor = tensor_util.MakeNdarray(
                    graph_info[graph_info[left_path_pad_names[0]].node.input[1]].node.
                    attr["value"].tensor).flatten()
                r_padding_tensor = tensor_util.MakeNdarray(
                    graph_info[graph_info[right_path_pad_names[0]].node.input[1]].node.
                    attr["value"].tensor).flatten()
                if not all(l_padding_tensor == r_padding_tensor):
                    excluded_pad.extend(left_path_pad_names)
                    excluded_pad.extend(right_path_pad_names)
            elif abs(len(left_path_pad_names) - len(right_path_pad_names)) == 1:
                    excluded_pad.extend(left_path_pad_names)
                    excluded_pad.extend(right_path_pad_names)
            else:
                pass
        # line 42 to line 83 should be removed once the tensorflow fix the inference shape issue.


        target_nodes = cur_graph.query_fusion_pattern_nodes(
            [["Pad"], ["Conv2D"], ('BiasAdd', 'Add')])

        for node_combination in target_nodes:
            pad_name = node_combination[0]
            conv_name = node_combination[1]
            pattern = node_combination[-1]
            is_perchannel = self.cfg[conv_name][0]

            # Line 98 to line 105 should be removed once the TFDO enabling the single quantized
            # conv2D supporting.
            if len(pattern) == 2:
                #TODO we need to enable single quantizedconv2d with s8 input.
                if not is_perchannel and not cur_graph.has_positive_input(conv_name):
                    continue
            # TFDO has the limitation that the single QuantizedConv2DPerchannel doesn't
            # support padding_list filed.
                if is_perchannel:
                    continue

            if pad_name in excluded_pad or conv_name in self.excluded_conv:
                continue

            pad_node = graph_info[node_combination[0]].node
            padding_tensor = tensor_util.MakeNdarray(
                graph_info[pad_node.input[1]].node.attr["value"].tensor).flatten()

            cur_graph.remove_node_with_single_input_output(pad_node.name)
            cur_graph.remove_node(pad_node.input[1])
            conv_node = graph_info[node_combination[1]].node
            Helper.set_attr_int_list(conv_node, "padding_list", padding_tensor)

        return cur_graph.dump_graph()
