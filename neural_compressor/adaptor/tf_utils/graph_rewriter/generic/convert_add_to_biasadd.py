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

import numpy as np
from tensorflow.python.framework import dtypes
from neural_compressor.utils.utility import dump_elapsed_time

from ..graph_base import GraphRewriterBase
from ..graph_util import GraphAnalyzer
from ..graph_util import GraphRewriterHelper as Helper
from tensorflow.python.framework import tensor_util

class ConvertAddToBiasAddOptimizer(GraphRewriterBase):
    """ Convert MatMul + Add(AddV2) to MatMul + BiasAdd.
    """
    @dump_elapsed_time("Pass ConvertAddToBiasAddOptimizer")
    def do_transformation(self):
        g = GraphAnalyzer()
        g.graph = self.model
        graph_info = g.parse_graph()

        target_nodes = g.query_fusion_pattern_nodes([['MatMul'], ['Add', 'AddV2']])
        for i in target_nodes:
            successor_node_names = graph_info[i[1]].outputs
            matmul_input_name = graph_info[i[0]].node.input[0]
            matmul_input_node = graph_info[Helper.node_name_from_input(matmul_input_name)].node
            #Fixme below two lines was added due to MatMul kernel limitation for matmul input type
            # should be quint8.
            if matmul_input_node.op == 'Const':
                continue
            add_second_input_name = graph_info[i[1]].node.input[1]
            add_second_const_node = graph_info[add_second_input_name].node
            if add_second_const_node.op != 'Const':
                continue
            bias_tensor = tensor_util.MakeNdarray(add_second_const_node.attr['value'].tensor)

            if bias_tensor.ndim > 2:
                continue

            new_bias_tensor = np.ravel(bias_tensor)

            g.remove_node(i[1])

            bias_node_name = i[1]
            bias_const_node_name = add_second_const_node.name + "_flattern"

            bias_const_node = Helper.create_constant_node(
                bias_const_node_name, new_bias_tensor, dtypes.float32)

            bias_node = Helper.create_node('BiasAdd', bias_node_name, [i[0], bias_const_node_name])
            Helper.set_attr_dtype(bias_node, "T", dtypes.float32)

            g.add_node(bias_const_node, None, [bias_node_name])
            g.replace_single_node(bias_node, [i[0]], i[1], successor_node_names,  i[1])
            
        return g.dump_graph()
