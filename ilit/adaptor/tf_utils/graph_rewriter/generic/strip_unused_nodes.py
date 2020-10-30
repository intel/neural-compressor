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


from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import attr_value_pb2

from ..graph_base import GraphRewriterBase
from ..graph_util import GraphAnalyzer


class StripUnusedNodesOptimizer(GraphRewriterBase):
    def __init__(self, model, input_node_names, output_node_names):
        super().__init__(model)
        self.input_node_names = input_node_names
        self.output_node_names = output_node_names

    def do_transformation(self):
        cur_graph = GraphAnalyzer()

        # according to https://github.com/onnx/tensorflow-onnx/issues/77
        for node in self.model.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr:
                    del node.attr['use_locking']
            elif node.op == 'AssignAdd':
                node.op = 'Add'
                if 'use_locking' in node.attr:
                    del node.attr['use_locking']
            elif node.op == 'Assign':
                node.op = 'Identity'
                if 'use_locking' in node.attr:
                    del node.attr['use_locking']
                if 'validate_shape' in node.attr:
                    del node.attr['validate_shape']
                if len(node.input) == 2:
                    # input0: ref: Should be from a Variable node. May be uninitialized.
                    # input1: value: The value to be assigned to the variable.
                    node.input[0] = node.input[1]
                    del node.input[1]

        cur_graph.graph = self.model

        graph_info = cur_graph.parse_graph()

        for name in self.input_node_names:
            if ':' in name:
                self.logger.debug("Name {} appears to refer to a Tensor, "
                                  "not a Operation.".format(name))
                return False

        type_attr = {"Sub": "T"}

        not_found = {name for name in self.input_node_names}
        for node_name, _ in graph_info.items():
            if node_name in not_found:
                not_found.remove(node_name)
                node = graph_info[node_name].node
                original_output = graph_info[node_name].outputs
                placeholder_node = node_def_pb2.NodeDef()
                placeholder_node.op = "Placeholder"
                placeholder_node.name = node.name

                if "dtype" in node.attr:
                    placeholder_node.attr["dtype"].CopyFrom(
                        attr_value_pb2.AttrValue(type=node.attr["dtype"].type))
                elif node.op in type_attr.keys():
                    placeholder_node.attr["dtype"].CopyFrom(
                        attr_value_pb2.AttrValue(type=node.attr[type_attr[node.op]].type))
                else:
                    raise KeyError("%s op's type attribute is not found,"
                                   "you should add it to type_attr dict" % node.op)
                if "_output_shapes" in node.attr:
                    placeholder_node.attr["_output_shapes"].CopyFrom(node.attr["_output_shapes"])
                if "shape" in node.attr:
                    placeholder_node.attr["shape"].CopyFrom(node.attr["shape"])

                cur_graph.remove_node(node_name)

                cur_graph.replace_const_node(placeholder_node, [node_name], original_output)

        import tensorflow as tf
        return tf.compat.v1.graph_util.extract_sub_graph(cur_graph.dump_graph(),
                                                         self.output_node_names)
