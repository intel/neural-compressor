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

import math
import re

import numpy as np
import tensorflow as tf
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import tf_logging
from ilit.adaptor.tf_utils.transform_graph.graph_transform_base import GraphTransformBase


class FoldConstant(GraphTransformBase):
    def __init__(self, input_graph_def):
        super().__init__(input_graph_def)
        self.supported_ops = ["Add", "AddV2", "Const", "Identity", "Mul", "Rsqrt", "Sub"]
        self.unfolded_ops = ["ConcatV2", "Concat", "Mean", "Reshape"]
        self.end_nodes = []
        self.unused_nodes = []

    def _fold_value(self, end_node_name):
        """calculate values of end node of constant node sequence

        there may bu constant node sequence in the graph, like:
          const -> add -> mul -> sqrt
        the value of sqrt can be calculated in advance.

        Args:
          end_node_name: name of the end node of the sequence. e.g. sqrt in the above examples.

        Returns:
          values of end node.

        Raises:
          ValueError: If the graph contains tensors which can't be broadcast.
        """
        end_node = self.input_node_map[end_node_name]

        def can_broadcast(s1, s2):
            s1a = np.asarray(s1)
            s2a = np.asarray(s2)
            return ((s1a == 1) | (s2a == 1) | (s2a == s1a)).all()

        if end_node.input:
            if end_node.op == "Mul":
                fold_value = np.array([1.])
                for index, input in enumerate(end_node.input):
                    # broadcast if needed
                    input_value = self._fold_value(input)
                    if can_broadcast(fold_value.shape, input_value.shape):
                        fold_value = fold_value * input_value
                    else:
                        raise ValueError(
                            "input {} of node {} can't be broadcast".format(
                                input.name, node.name))
                return fold_value
            elif end_node.op == "Add" or end_node.op == "AddV2":
                fold_value = np.array([0.])
                for index, input in enumerate(end_node.input):
                    # broadcast if needed
                    input_value = self._fold_value(input)
                    if can_broadcast(fold_value.shape, input_value.shape):
                        fold_value = fold_value + input_value
                    else:
                        raise ValueError(
                            "input {} of node {} can't be broadcast".format(
                                input.name, end_node.name))
                return fold_value
            elif end_node.op == "Rsqrt":
                return 1/np.sqrt(self._fold_value(end_node.input[0]))
            elif end_node.op == "Identity":
                return self._fold_value(end_node.input[0])
            elif end_node.op == "Sub":
                fold_value = np.array([0.])
                for index, input in enumerate(end_node.input):
                    # broadcast if needed
                    input_value = self._fold_value(input)
                    if can_broadcast(fold_value.shape, input_value.shape):
                        fold_value = fold_value + (-1) ** index * input_value
                    else:
                        raise ValueError(
                            "input {} of node {} can't be broadcast".format(
                                input.name, end_node.name))
                return fold_value
            else:
                tf_logging.info(
                    "Currently fold-constant only support limited ops {} but face {}".format(self.supported_ops, end_node.op))
        else:
            return self.values_from_const(end_node)

    def _recursive_get_fold_end(self, node):
        """helper function of get all end nodes
        """
        constant_node_flag = True
        if node.name in self.contant_node_map:
            return self.contant_node_map[node.name]
        sub_end_nodes = []
        input_nodes = list(node.input)
        for input_node_name in node.input:
            input_node = self.input_node_map[input_node_name]
            if input_node.op == "Identity" :
                constant_sub_node_flag = (len(input_node.input) == 1 and self.input_node_map[input_node.input[0]].op == "Const")
            elif input_node.op == "Const":
                constant_sub_node_flag = True
            elif input_node.op in self.unfolded_ops:
                self._recursive_get_fold_end(input_node)
                constant_sub_node_flag = False
            else:
                constant_sub_node_flag = self._recursive_get_fold_end(input_node)
                if constant_sub_node_flag:
                    if input_node not in self.unused_nodes:
                        self.unused_nodes.append(input_node)
                    sub_end_nodes.append(input_node.name)
            constant_node_flag &= constant_sub_node_flag
        if not constant_node_flag or node.op not in self.supported_ops:
            for input_node in input_nodes:
                if input_node in sub_end_nodes:
                    self.contant_node_map[input_node] = True
                    self.end_nodes.append(input_node)
                else:
                    self.contant_node_map[input_node] = False
        return constant_node_flag and node.op in self.supported_ops

    def _get_fold_end(self, output_node_name):
        """get all end nodes of constant node sequence

        the function will search all the parent nodes of the input node recursively and check if the node is an end node of a constant sequence

        Args:
          node: name of a node, at beginning it is the predict node of the whole graph.

        Returns:
          boolen: if the node is an end node of a constant sequence

        """
        self.contant_node_map = {}
        output_node = self.input_node_map[output_node_name]
        self._recursive_get_fold_end(output_node)

    def do_transformation(self, input_nodes_name, output_nodes_name):
        """fold all the sequences only consist of const and self.supported_ops

        Args:
          input_nodes_name: list of names of input_nodes of the whole graph
          output_nodes_name: list of names of output_nodes of the whole graph

        Returns:
           Modified graph with constant sequence removed, and modified weights.

        """
        result_graph = graph_pb2.GraphDef()
        new_end_nodes = []
        for output_node_name in output_nodes_name:
            self._get_fold_end(output_node_name)
        for end_node_name in self.end_nodes:
            
            end_node = self.input_node_map[end_node_name]
            new_end_node = node_def_pb2.NodeDef()
            new_end_node.op = "Const"
            new_end_node.name = end_node.name
            new_end_value = np.float32(self._fold_value(end_node.name))
            new_end_value_type = tf.as_dtype(new_end_value.dtype)
            new_end_node.attr["dtype"].CopyFrom(end_node.attr["T"])
            new_end_node.attr["value"].CopyFrom(attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
                new_end_value, new_end_value_type, new_end_value.shape)))
            new_end_nodes.append(new_end_node)

        for node in self.input_graph.node:
            if node in self.unused_nodes:
                continue
            else:
                result_graph.node.append(node)
        result_graph.node.extend(new_end_nodes)

        return result_graph
