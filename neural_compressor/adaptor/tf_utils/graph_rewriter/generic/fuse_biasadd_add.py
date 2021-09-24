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

import tensorflow as tf
from ..graph_base import GraphRewriterBase
from ..graph_util import GraphAnalyzer
from ..graph_util import GraphRewriterHelper as Helper
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util


class FuseBiasAddAndAddOptimizer(GraphRewriterBase):
    """Fuse Biasadd + Add into BiasAdd when the second input of Add is const node.
    """

    def do_transformation(self):

        cur_graph = GraphAnalyzer()
        cur_graph.graph = self.model

        graph_info = cur_graph.parse_graph()

        target_nodes = cur_graph.query_fusion_pattern_nodes(
            [["Conv2D"], "BiasAdd", ["Add", "AddV2"], ["Relu", "Relu6"]])

        for i in target_nodes:
            biasadd_const_name = graph_info[i[1]].node.input[1]
            biasadd_const_node = graph_info[biasadd_const_name].node

            if len(graph_info[i[1]].outputs) > 1:
                continue

            another_node_index = None
            for index, value in enumerate(graph_info[i[2]].node.input):
                if value != i[1]:
                    another_node_index = index
                    break
            add_node_const_name = graph_info[i[2]].node.input[another_node_index]

            add_const_node = graph_info[add_node_const_name].node

            if add_const_node.op != 'Const':
                continue
            value= tensor_util.MakeNdarray(
                                biasadd_const_node.attr['value'].tensor)
            add_value =  tensor_util.MakeNdarray(
                                add_const_node.attr['value'].tensor)

            new_bias_tensor = (value+add_value)
            fused_const_node = Helper.create_constant_node(
                        i[2]+'_fused', new_bias_tensor, dtypes.float32)
            cur_graph.remove_node(graph_info[i[1]].node.input[1])

            graph_info[i[1]].node.input[1] = i[2] + '_fused'

            cur_graph.remove_node(add_node_const_name)

            cur_graph.remove_node(i[2])
            graph_info[i[3]].node.input[0] = i[1]
            cur_graph.add_node(fused_const_node, None, [i[1]])

        return cur_graph.dump_graph()
