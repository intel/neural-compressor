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
        self.supported_node = {}
        self.end_nodes = set()
        self.unused_nodes = []
        self.generate_input_map()

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
        end_node = self.input_node_map[self.node_name_from_input(end_node_name)]

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
                                input.name, end_node.name))
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
                return 1 / np.sqrt(self._fold_value(end_node.input[0]))
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

    def _recursive_get_support_node(self, node_name):
        """helper function of get all support nodes based on constant-root
        """
        if node_name in self.supported_node:
            return self.supported_node[node_name]
        node = self.input_node_map[node_name]
        if node.input:
            input_nodes = list(node.input)
            support_node_flag = node.op in self.supported_ops
            for input_node_name in input_nodes:
                input_node_name = self.node_name_from_input(input_node_name)
                support_node_flag &= self._recursive_get_support_node(input_node_name)
            self.supported_node[node_name] = support_node_flag
            return support_node_flag
        else:
            self.supported_node[node_name] = node.op == "Const"
            return node.op == "Const"

    def generate_output_map(self, output_node_name):
        """generate maps of node_name->output_node_name
        """
        output_node_name = self.node_name_from_input(output_node_name)
        if self.input_node_map[output_node_name].input:
            for node in self.input_node_map[output_node_name].input:
                node = self.node_name_from_input(node)
                if node in self.output_node_map:
                    self.output_node_map[node].add(output_node_name)
                    continue
                else:
                    self.output_node_map[node] = {output_node_name}
                self.generate_output_map(node)
        else:
            return

    def generate_input_map(self):
        """generate maps of node_name->node
        """
        self.input_node_map = {}
        for node in self.input_graph.node:
            node_name = self.node_name_from_input(node.name)
            if node_name not in self.input_node_map:
                self.input_node_map[node_name] = node
            else:
                raise ValueError("Duplicate node names detected for ",
                                 node.name)

    def _get_constant_node_map(self, input_nodes_name, output_nodes_name):
        """input_nodes like image_tensor are non-constant, so their outpus are non-constant. this function helps to find all constant nodes.

        Args:
          input_nodes_name: list of names of input_nodes of the whole graph
          output_nodes_name: list of names of output_nodes of the whole graph

        Returns:
          None
        """
        for output_node_name in output_nodes_name:
            self.generate_output_map(output_node_name)
        input_nodes_name = set(input_nodes_name)
        non_constant_nodes = output_nodes_name = set(output_nodes_name)
        current_nodes = input_nodes_name
        tmp_nodes = set()
        while current_nodes:
            for input_node_name in list(current_nodes):
                input_node_name = self.node_name_from_input(input_node_name)
                if input_node_name in non_constant_nodes:
                    continue
                non_constant_nodes.add(input_node_name)
                for out_node in self.output_node_map[input_node_name]:
                    tmp_nodes.add(out_node)

            current_nodes = tmp_nodes
            tmp_nodes = set()
        self.constant_nodes = [
            self.node_name_from_input(
                node.name) for node in list(
                self.input_graph.node) if self.node_name_from_input(
                node.name) not in non_constant_nodes and node.input]

    def _check_end(self, node_name):
        """if the node is constant and one of its output is non-constant, the node can possiblely be end of sequence.

        Args:
          node_name: name of a node

        Returns:
          boolen: if the node is an end node of a constant sequence
        """
        constant_flag = True
        node = self.input_node_map[node_name]
        for input_node_name in self.output_node_map[node_name]:
            constant_flag &= node in self.constant_nodes
        return not constant_flag and node_name in self.constant_nodes

    def _get_fold_end(self):
        """get all nodes of constant node sequence including ends nodes.

        the function will search all the constant nodes to check if it is part of sequence.
        """
        for node in self.input_graph.node:
            node_name = self.node_name_from_input(node.name)
            if node_name in self.output_node_map:
                if self._check_end(node_name):
                    self._recursive_get_support_node(node_name)
        for node_name in self.supported_node:
            if self.supported_node[node_name]:
                node = self.input_node_map[node_name]
                support_flag = True
                for output_node_name in self.output_node_map[node_name]:
                    if output_node_name in self.supported_node:
                        support_flag &= self.supported_node[output_node_name]
                    else:
                        support_flag = False
                if not support_flag:
                    self.end_nodes.add(node_name)
                if node.op == "Identity":
                    output_node_name = list(self.output_node_map[node_name])[0]
                    if output_node_name in self.constant_nodes:
                        self.unused_nodes.append(node)
                elif node.op == "Const":
                    continue
                else:
                    self.unused_nodes.append(node)
        for node_name in self.supported_node:
            node = self.input_node_map[node_name]
            if node.op == "Const":
                output_node_name = list(self.output_node_map[node_name])[0]
                output_node = self.input_node_map[output_node_name]
                if output_node in self.unused_nodes:
                    self.unused_nodes.append(node)

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
        self._get_constant_node_map(input_nodes_name, output_nodes_name)
        self._get_fold_end()

        for end_node_name in self.end_nodes:
            if self.input_node_map[end_node_name].op == "Const" or self.input_node_map[end_node_name].op == "Identity":
                continue

            end_node = self.input_node_map[self.node_name_from_input(end_node_name)]
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
