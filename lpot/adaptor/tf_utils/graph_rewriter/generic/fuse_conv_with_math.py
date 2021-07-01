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
from tensorflow.python.framework import tensor_util
from lpot.utils.utility import dump_elapsed_time

from ..graph_base import GraphRewriterBase
from ..graph_util import GraphAnalyzer
from ..graph_util import GraphRewriterHelper as Helper


class FuseConvWithMathOptimizer(GraphRewriterBase):
    """ Convert below subgraph to Conv2D + BiasAdd by eliminating math ops.
                Conv2D            Conv2D
                  |                  |
                 Sub                 |
                  |       ---->      |
                RealDiv              |
                  |                  |
                 Mul                 |
                  |                  |
                BiasAdd           BiasAdd
    """
    @dump_elapsed_time("Pass FuseConvWithMathOptimizer")
    def do_transformation(self):
        g = GraphAnalyzer()
        g.graph = self.model
        graph_info = g.parse_graph()
        pattern_definition = [['Conv2D'], ['Sub'], ['RealDiv'], ['Mul'], ['BiasAdd']]
        target_nodes = g.query_fusion_pattern_nodes(pattern_definition)
        for i in target_nodes:
            weights_node_name = graph_info[i[0]].node.input[1]
            weights_node = graph_info[weights_node_name].node

            sub_input_names = list(graph_info[i[1]].node.input)
            sub_content_node_name = list(set(sub_input_names).difference([i[0]]))[0]
            sub_content_node = graph_info[sub_content_node_name].node
            sub_tensor = tensor_util.MakeNdarray(sub_content_node.attr['value'].tensor)

            real_div_input_names = list(graph_info[i[2]].node.input)
            real_div_content_node_name = list(set(real_div_input_names).difference([i[1]]))[0]
            real_div_node = graph_info[real_div_content_node_name].node
            real_div_tensor = tensor_util.MakeNdarray(real_div_node.attr['value'].tensor)

            mul_input_names = list(graph_info[i[3]].node.input)
            mul_content_node_name = list(set(mul_input_names).difference([i[2]]))[0]
            mul_content_node = graph_info[mul_content_node_name].node
            mul_tensor = tensor_util.MakeNdarray(mul_content_node.attr['value'].tensor)

            bias_input_names = list(graph_info[i[4]].node.input)
            bias_content_node_name = list(set(bias_input_names).difference([i[3]]))[0]
            bias_content_node = graph_info[bias_content_node_name].node
            bias_tensor = tensor_util.MakeNdarray(bias_content_node.attr['value'].tensor)

            bias_offset_value = bias_tensor - sub_tensor*mul_tensor / real_div_tensor
            weights_offset = mul_tensor / real_div_tensor

            weights = Helper.values_from_const(weights_node)
            original_shape = weights.shape
            tmp_shape = (original_shape[-1], int(weights.size/original_shape[-1]))
            tmp_order = [weights.ndim - 1] + [i for i in range(weights.ndim - 1)]

            scaled_weights = np.copy(weights).transpose(tmp_order).ravel().reshape(tmp_shape)
            reshape_scale = np.array(weights_offset).reshape(len(weights_offset), 1)
            scaled_weights = np.multiply(
                scaled_weights, reshape_scale).transpose().reshape(original_shape)
            scaled_weight_name = weights_node_name + "_conv_math_offset"
            scaled_weights_node = Helper.create_constant_node(scaled_weight_name,
                                    scaled_weights, dtypes.float32, shape=weights.shape)

            g.add_node(scaled_weights_node, None, [i[0]])
            g.replace_const_node(scaled_weights_node, [i[0]], weights_node_name)

            offset_node = Helper.create_constant_node(i[0] + "_biasadd_math_offset",
                                    bias_offset_value, dtypes.float32)
            g.add_node(offset_node, None, [i[4]])
            graph_info[i[4]].node.input[0] = i[0]

            graph_info[i[4]].node.input[1] = offset_node.name

            g.remove_node(i[1])
            g.remove_node(sub_content_node_name)

            g.remove_node(i[2])
            g.remove_node(real_div_content_node_name)

            g.remove_node(i[3])
            g.remove_node(mul_content_node_name)

            g.remove_node(bias_content_node_name)

        return g.dump_graph()
