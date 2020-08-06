#
#  -*- coding: utf-8 -*-
#
#  Copyright (c) 2020 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import graph_util

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import node_def_pb2
from .graph_transform_base import GraphTransformBase

import tensorflow as tf
import copy


class StripUnusedNodes(GraphTransformBase):
    def __init__(self, input_graph_def, input_node_names, output_node_names):
        self.input_node_names = input_node_names
        self.output_node_names = output_node_names
        self.input_graph = input_graph_def

    def do_transform(self):
        """Removes unused nodes from a GraphDef.

        Args:
          input_graph_def: A graph with nodes we want to prune.
          input_node_names: A list of the nodes we use as inputs.
          output_node_names: A list of the output nodes.

        Returns:
          A `GraphDef` with all unnecessary ops removed.

        Raises:
          ValueError: If any element in `input_node_names` refers to a tensor instead
            of an operation.
          KeyError: If any element in `input_node_names` is not found in the graph.
        """
        for name in self.input_node_names:
            if ":" in name:
                raise ValueError("Name '%s' appears to refer to a Tensor, "
                                 "not a Operation." % name)

        # Here we replace the nodes we're going to override as inputs with
        # placeholders so that any unused nodes that are inputs to them are
        # automatically stripped out by extract_sub_graph().
        not_found = {name for name in self.input_node_names}
        inputs_replaced_graph_def = graph_pb2.GraphDef()
        for node in self.input_graph.node:
            if node.name in self.input_node_names:
                not_found.remove(node.name)
                placeholder_node = node_def_pb2.NodeDef()
                placeholder_node.op = "Placeholder"
                placeholder_node.name = node.name
                placeholder_node.attr["dtype"].CopyFrom(
                        attr_value_pb2.AttrValue(type=node.attr["dtype"].type))
                if "_output_shapes" in node.attr:
                    placeholder_node.attr["_output_shapes"].CopyFrom(
                        node.attr["_output_shapes"])
                if "shape" in node.attr:
                    placeholder_node.attr["shape"].CopyFrom(node.attr["shape"])
                inputs_replaced_graph_def.node.extend([placeholder_node])
            else:
                inputs_replaced_graph_def.node.extend([copy.deepcopy(node)])

        if not_found:
            raise KeyError("The following input nodes were not found: %s" %
                           not_found)

        output_graph_def = tf.compat.v1.graph_util.extract_sub_graph(
            inputs_replaced_graph_def, self.output_node_names)
        return output_graph_def
