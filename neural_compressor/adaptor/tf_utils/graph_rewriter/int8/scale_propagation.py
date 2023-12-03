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
"""Scale propagation Graph Rewriter."""

from tensorflow.core.framework import attr_value_pb2, node_def_pb2
from tensorflow.python.framework import dtypes, tensor_util

from neural_compressor.adaptor.tf_utils.graph_util import GraphAnalyzer
from neural_compressor.adaptor.tf_utils.graph_util import GraphRewriterHelper as Helper

from ..graph_base import GraphRewriterBase


class ScaleProPagationTransformer(GraphRewriterBase):
    """Scale propagation converter."""

    cac_pattern = [
        ["QuantizeV2", "Requantize", "RequantizePerChannel"],
        ["QuantizedAvgPool"],
        [
            "QuantizedConv2DWithBias",
            "QuantizedConv2DWithBiasAndRelu",
            "QuantizedConv2DPerChannel",
            "QuantizedConv2D",
            "QuantizedConv2DWithBiasSumAndRelu",
        ],
        ["Requantize", "RequantizePerChannel"],
    ]

    def __init__(self, model, direction="Up"):
        """Initialization."""
        super().__init__(model)
        self.direction = direction if direction not in ("Up", "Down") else "Up"
        self.cur_graph = GraphAnalyzer()
        self.cur_graph.graph = self.model

        self.graph_info = self.cur_graph.parse_graph()

    def _create_new_const_node(self, new_const_node_name, value, old_const_node_name):
        """Create new Const node."""
        new_node = node_def_pb2.NodeDef()
        new_node.op = "Const"
        new_node.name = new_const_node_name
        new_node.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
        new_node.attr["value"].CopyFrom(
            attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(float(value), dtypes.float32, []))
        )
        output_node_name = self.graph_info[old_const_node_name].outputs[0]
        self.cur_graph.replace_const_node(
            new_node, [Helper.node_name_from_input(output_node_name)], old_const_node_name
        )
        self.cur_graph.remove_node(old_const_node_name)

    def _cac_transformation(self):
        """Scale propagation pattern match and create new const nodes."""
        target_nodes = self.cur_graph.query_fusion_pattern_nodes(self.cac_pattern)
        quantize_v2_min_index = 1
        requntize_min_index = 3
        quantize_v2_max_index = 2
        requntize_max_index = 4

        for match in target_nodes:
            pre_node_name = match[0]

            pre_node = self.graph_info[pre_node_name].node

            output_nodes_count = len(set(self.graph_info[pre_node_name].outputs))

            if output_nodes_count > 1:
                continue
            # Skip transformation if avgpool has multi output nodes.
            pooling_nodes_count = len(set(self.graph_info[match[1]].outputs))
            if pooling_nodes_count > 1:
                continue

            if pre_node.op == "QuantizeV2":
                pre_min_index, pre_max_index = quantize_v2_min_index, quantize_v2_max_index
            else:
                pre_min_index, pre_max_index = requntize_min_index, requntize_max_index

            requantize_node_name = match[3]
            requantize_node = self.graph_info[requantize_node_name].node

            requantize_min = self.graph_info[
                Helper.node_name_from_input(requantize_node.input[requntize_min_index])
            ].node
            requantize_max = self.graph_info[
                Helper.node_name_from_input(requantize_node.input[requntize_max_index])
            ].node

            requantize_min_value = (requantize_min.attr["value"].tensor.float_val)[0]
            requantize_max_value = (requantize_max.attr["value"].tensor.float_val)[0]
            self._create_new_const_node(
                pre_node_name + "_cac_requantize_min_value", requantize_min_value, pre_node.input[pre_min_index]
            )
            self._create_new_const_node(
                pre_node_name + "_cac_requantize_max_value", requantize_max_value, pre_node.input[pre_max_index]
            )

    def do_transformation(self):
        """Apply the scale propagation algrothim."""
        self._cac_transformation()

        return self.cur_graph.dump_graph()
