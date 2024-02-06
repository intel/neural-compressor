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
"""Fuse Columnwise Mul Graph Rewriter."""

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import dtypes, tensor_util

from neural_compressor.tensorflow.quantization.utils.graph_util import GraphAnalyzer
from neural_compressor.tensorflow.quantization.utils.graph_util import GraphRewriterHelper as Helper
from neural_compressor.tensorflow.utils import dump_elapsed_time

from ..graph_base import GraphRewriterBase


class FuseColumnWiseMulOptimizer(GraphRewriterBase):
    """Fuse Mul op into Conv2D/DepthwiseConv2dNative/MatMul."""

    @dump_elapsed_time("Pass FuseColumnWiseMulOptimizer")
    def do_transformation(self):
        """Fuse Mul + Conv2D/DepthwiseConv2dNative/MatMul --> Conv2D/DepthwiseConv2dNative/MatMul."""
        cur_graph = GraphAnalyzer()
        cur_graph.graph = self.model

        graph_info = cur_graph.parse_graph()
        target_nodes = cur_graph.query_fusion_pattern_nodes([["Conv2D", "DepthwiseConv2dNative", "MatMul"], "Mul"])

        for node_combination in target_nodes:
            upper_node = graph_info[node_combination[0]].node
            mul_node = graph_info[node_combination[1]].node
            if graph_info[Helper.node_name_from_input(mul_node.input[1])].node.op != "Const":
                continue
            weights_node = graph_info[graph_info[node_combination[0]].node.input[1]].node
            mul_value_node = graph_info[graph_info[node_combination[1]].node.input[1]].node
            upper_node_type = upper_node.op

            if upper_node_type == "Conv2D":
                weights_col = weights_node.attr["value"].tensor.tensor_shape.dim[3].size
            elif upper_node_type == "DepthwiseConv2dNative":
                weights_col = (
                    weights_node.attr["value"].tensor.tensor_shape.dim[2].size
                    * weights_node.attr["value"].tensor.tensor_shape.dim[3].size
                )
            else:
                weights_col = weights_node.attr["value"].tensor.tensor_shape.dim[1].size

            mul_value_node_tensor = mul_value_node.attr["value"].tensor
            weights_node_tensor = weights_node.attr["value"].tensor
            if (
                len(mul_value_node_tensor.tensor_shape.dim) != 1
                or mul_value_node_tensor.tensor_shape.dim[0].size != weights_col
            ):
                self.logger.warning("Invalid Mul OP fusion.")
                return self.model

            mul_value_node_list = [i for i in tensor_util.MakeNdarray(mul_value_node_tensor).flat]
            new_weights = []
            for index, i in enumerate(tensor_util.MakeNdarray(weights_node_tensor).flat):
                new_weights_value = i * mul_value_node_list[index % len(mul_value_node_list)]
                new_weights.append(new_weights_value)

            weights_node.attr["value"].CopyFrom(
                attr_value_pb2.AttrValue(
                    tensor=tensor_util.make_tensor_proto(
                        new_weights, dtypes.float32, tensor_util.MakeNdarray(weights_node_tensor).shape
                    )
                )
            )

            cur_graph.remove_node_with_single_input_output(mul_node.name)
            cur_graph.remove_node(mul_node.input[1])

        return cur_graph.dump_graph()
