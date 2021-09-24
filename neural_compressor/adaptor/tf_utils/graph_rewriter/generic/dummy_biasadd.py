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

from tensorflow.python.framework import dtypes
from neural_compressor.utils.utility import dump_elapsed_time

from ..graph_base import GraphRewriterBase
from ..graph_util import GraphAnalyzer
from ..graph_util import GraphRewriterHelper as Helper


class InjectDummyBiasAddOptimizer(GraphRewriterBase):
    @dump_elapsed_time("Pass InjectDummyBiasAddOptimizer")
    def do_transformation(self):
        g = GraphAnalyzer()
        g.graph = self.model
        graph_info = g.parse_graph()
        valid_ops = ('BiasAdd', 'Add', 'AddV2', 'AddN')
        target_nodes = g.query_fusion_pattern_nodes([['MatMul'],])
        for i in target_nodes:
            next_node_names = graph_info[i[0]].outputs
            if next_node_names and len(next_node_names) == 1 and \
                graph_info[Helper.node_name_from_input(next_node_names[0])].node.op in valid_ops:
                continue
            bias_node_name = i[0] + '_dummy_biasadd'
            bias_const_node_name = i[0] + '_fake_const'
            next_node_names = graph_info[i[0]].outputs
            matmul_b_node_name = graph_info[i[0]].node.input[1]
            matmul_b_node = graph_info[matmul_b_node_name].node
            exit_cur_loop = False
            while matmul_b_node.op != 'Const':
                last_node = graph_info[matmul_b_node_name].node
                if last_node.input and last_node.op != 'Enter':
                    matmul_b_node_name = last_node.input[0]
                else:
                    exit_cur_loop = True
                    break
                matmul_b_node = graph_info[matmul_b_node_name].node

            if exit_cur_loop:
                continue

            t_b_index = 0 if graph_info[i[0]].node.attr['transpose_b'].b else 1
            bias_add_length = matmul_b_node.attr['value'].tensor.tensor_shape.dim[t_b_index].size

            bias_add_content = [0.] * bias_add_length

            bias_const_node = Helper.create_constant_node(
                bias_const_node_name, bias_add_content, dtypes.float32, shape=[bias_add_length])
            bias_node = Helper.create_node('BiasAdd', bias_node_name, [i[0], bias_const_node_name])
            Helper.set_attr_dtype(bias_node, "T", dtypes.float32)
            g.add_node(bias_node, i[0], next_node_names)
            g.add_node(bias_const_node, None, [bias_node_name])

        return g.dump_graph()
