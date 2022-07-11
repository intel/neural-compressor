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
from tensorflow.python.framework import dtypes
from neural_compressor.utils.utility import dump_elapsed_time

from ..graph_base import GraphRewriterBase
from neural_compressor.adaptor.tf_utils.graph_util import GraphAnalyzer
from neural_compressor.adaptor.tf_utils.graph_util import GraphRewriterHelper as Helper
from neural_compressor.adaptor.tf_utils.util import version1_gt_version2


class InjectDummyBiasAddOptimizer(GraphRewriterBase):
    def __init__(self, model, outputs):
        super().__init__(model)
        self.outputs = outputs

    @dump_elapsed_time("Pass InjectDummyBiasAddOptimizer")
    def do_transformation(self):
        g = GraphAnalyzer()
        g.graph = self.model
        graph_info = g.parse_graph()
        valid_ops = ('BiasAdd', 'Add', 'AddV2', 'AddN')
        target_nodes = g.query_fusion_pattern_nodes([['MatMul', 'Conv2D'],])
        for i in target_nodes:
            # only apply this pass for tensorflow release 2.9.1 and lower version for
            # old quantization API.
            # use conv+dummy_biasadd+relu because TF do not support conv+relu now. 
            if version1_gt_version2(tf.version.VERSION, '2.9.1'):
                continue
            if i[0] in self.outputs:
                continue
            next_node_names = graph_info[i[0]].outputs
            if next_node_names and len(next_node_names) == 1 and \
                graph_info[Helper.node_name_from_input(next_node_names[0])].node.op in valid_ops:
                continue
            bias_node_name = i[0] + '_dummy_biasadd'
            bias_const_node_name = i[0] + '_fake_const'
            matmul_a_node_name = Helper.node_name_from_input(graph_info[i[0]].node.input[0])
            matmul_a_node = graph_info[matmul_a_node_name].node
            matmul_b_node_name = Helper.node_name_from_input(graph_info[i[0]].node.input[1])
            matmul_b_node = graph_info[matmul_b_node_name].node

            if matmul_a_node.op == 'Const' or matmul_b_node.op != 'Const':
                continue

            if graph_info[i[0]].node.op == 'MatMul':
                t_b_index = 0 if graph_info[i[0]].node.attr['transpose_b'].b else 1
            elif graph_info[i[0]].node.op == 'Conv2D' and graph_info[i[0]].node.attr['data_format'].s == b'NHWC':
                t_b_index = 3
            elif graph_info[i[0]].node.op == 'Conv2D' and graph_info[i[0]].node.attr['data_format'].s == b'NCHW':
                t_b_index = 1
            else:
                continue

            bias_add_length = matmul_b_node.attr['value'].tensor.tensor_shape.dim[t_b_index].size

            bias_add_content = [0.] * bias_add_length

            bias_const_node = Helper.create_constant_node(
                bias_const_node_name, bias_add_content, dtypes.float32, shape=[bias_add_length])
            bias_node = Helper.create_node('BiasAdd', bias_node_name, [i[0], bias_const_node_name])
            Helper.set_attr_dtype(bias_node, "T", dtypes.float32)
            g.add_node(bias_node, i[0], next_node_names)
            g.add_node(bias_const_node, None, [bias_node_name])

        return g.dump_graph()
