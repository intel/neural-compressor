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
"""Convert placeholder to const Graph Rewriter."""

from tensorflow.core.framework import attr_value_pb2, node_def_pb2
from tensorflow.python.framework import dtypes, tensor_util

from neural_compressor.tensorflow.quantization.utils.graph_util import GraphAnalyzer
from neural_compressor.tensorflow.quantization.utils.graph_util import GraphRewriterHelper as Helper
from neural_compressor.tensorflow.utils import dump_elapsed_time

from ..graph_base import GraphRewriterBase


class ConvertPlaceholderToConst(GraphRewriterBase):
    """Convert placeholder to const for removing training nodes."""

    @dump_elapsed_time("Pass ConvertPlaceholderToConst")
    def do_transformation(self):
        """Rename the PlaceHolderWithDefault node to constant.

        In a frozen graph, PlaceholderWithDefault nodes can be converted to
        Constant op nodes with same value. This will help simplify the graph.

        Args:
            input_graph_def: A GraphDef containing a model.
            nodes_to_convert: A list of PlaceholderWithDefault or Placeholder
            nodes to be converted to Constants with their new value.

        Returns:
            modified graph with PlaceholderWithDefault node converted to Constant node
        """
        cur_graph = GraphAnalyzer()
        cur_graph.graph = self.model

        graph_info = cur_graph.parse_graph()

        target_nodes = cur_graph.query_fusion_pattern_nodes([["PlaceholderWithDefault"]])
        for i in target_nodes:
            placeholder_node = graph_info[i[0]].node
            new_node = node_def_pb2.NodeDef()
            if dtypes.bool.as_datatype_enum == placeholder_node.attr["dtype"].type:
                placeholder_input_node = None
                if placeholder_node.input:
                    placeholder_input_node = graph_info[Helper.node_name_from_input(placeholder_node.input[0])].node

                if placeholder_input_node and placeholder_input_node.op != "Const":
                    continue
                if placeholder_input_node:
                    new_val_str = placeholder_input_node.attr["value"].tensor.bool_val
                else:
                    continue

                new_node.op = "Const"
                new_node.name = placeholder_node.name + "_const"
                new_node.attr["dtype"].CopyFrom(placeholder_node.attr["dtype"])
                new_node.attr["value"].CopyFrom(
                    attr_value_pb2.AttrValue(
                        tensor=tensor_util.make_tensor_proto(self.strtobool(new_val_str), dtype=dtypes.bool, shape=[])
                    )
                )
                cur_graph.add_node(new_node, None, graph_info[placeholder_node.name].outputs)
                for each_output in graph_info[placeholder_node.name].outputs:
                    for i, input_name in enumerate(graph_info[each_output].node.input):
                        if input_name == placeholder_node.name:
                            new_input = (
                                graph_info[each_output].node.input[:i]
                                + [new_node.name]
                                + graph_info[each_output].node.input[i + 1 :]
                            )
                            graph_info[each_output].node.ClearField("input")
                            graph_info[each_output].node.input.extend(new_input)
                cur_graph.remove_node(placeholder_node.name)
            else:
                continue

        return cur_graph.dump_graph()

    def strtobool(self, val_str):
        """Return boolean value of it's equivalent string representation."""
        if val_str == [True]:
            return True
        if val_str == [False]:
            return False
        return False
