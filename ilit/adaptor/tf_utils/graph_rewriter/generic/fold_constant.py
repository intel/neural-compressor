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


import numpy as np
import tensorflow as tf

from tensorflow.python.platform import tf_logging
from ..graph_base import GraphRewriterBase
from ..graph_util import GraphAnalyzer, GraphRewriterHelper


class GraphFoldConstantOptimizer(GraphRewriterBase):
    supported_op_type = ["Add", "AddV2", "Const", "Mul", "Rsqrt", "Sub"]

    def __init__(self, model=None):
        super().__init__(model)
        self.graph_analyzer = GraphAnalyzer()
        self.graph_analyzer.graph = self.model

        self.graph_info = self.graph_analyzer.parse_graph()

    def _fold_value(self, end_node_name):
        """calculate values of end node of constant node sequence

        there may be layers whose inputs are all constant in the graph, like:
          const
                > add
          const
        the value of add can be calculated in advance.

        Args:
          end_node_name: name of the end node of the sequence. e.g. add in the above examples.

        Returns:
          values of end node.

        Raises:
          ValueError: If the graph contains tensors which can't be broadcast.
        """

        end_node = self.graph_info[end_node_name].node

        def can_broadcast(s1, s2):
            if s1.shape and s2.shape:
                s1a = np.asarray(s1.shape)
                s2a = np.asarray(s2.shape)
                return ((s1a == 1) | (s2a == 1) | (s2a == s1a)).all()
            else:
                return True

        if self.graph_info[end_node_name].node.input:
            if end_node.op == "Mul":
                first_value = self._fold_value(list(end_node.input)[0])
                first_type = first_value.dtype
                fold_value = np.array(1.).astype(first_type)
                for index, input in enumerate(end_node.input):
                    # broadcast if needed
                    input_value = self._fold_value(input)
                    input_type = input_value.dtype
                    if can_broadcast(fold_value, input_value):
                        fold_value = fold_value * input_value
                    else:
                        raise ValueError("input {} of node {} can't be broadcast".format(
                            input.name, end_node.name))
                return fold_value.astype(first_type)
            elif end_node.op == "Add" or end_node.op == "AddV2":
                first_value = self._fold_value(list(end_node.input)[0])
                first_type = first_value.dtype
                fold_value = np.array(0.).astype(first_type).reshape(())
                for index, input in enumerate(end_node.input):
                    # broadcast if needed
                    input_value = self._fold_value(input)
                    if can_broadcast(fold_value, input_value):
                        fold_value = fold_value + input_value
                    else:
                        raise ValueError("input {} of node {} can't be broadcast".format(
                            input.name, end_node.name))
                return fold_value.astype(first_type)
            elif end_node.op == "Rsqrt":
                return 1 / np.sqrt(self._fold_value(end_node.input[0]))
            elif end_node.op == "Sub":
                first_value = self._fold_value(list(end_node.input)[0])
                first_type = first_value.dtype
                fold_value = np.array(0., dtype=first_type)
                for index, input in enumerate(end_node.input):
                    # broadcast if needed
                    input_value = self._fold_value(input)
                    if first_type != input_value.dtype:
                        raise ValueError(
                            "input of node {} must be in same dtype but get {}and {}".format(
                                input.name, first_type, input_value.dtype))
                    if can_broadcast(fold_value, input_value):
                        fold_value = fold_value + (-1)**index * input_value
                    else:
                        raise ValueError("input {} of node {} can't be broadcast".format(
                            input.name, end_node.name))
                return fold_value.astype(first_type)
            else:
                tf_logging.info(
                    "Currently fold-constant only support limited ops {} but face {}".format(
                        self.supported_op_type, end_node.op))
        else:
            return GraphRewriterHelper.values_from_const(end_node)

    def check_all_folded(self):
        """Check the node has been folded completely.

        Returns:
            bool: True if the node has been folded else False.
        """
        for node_name, _ in self.graph_info.items():
            if self.check_const_inputs(node_name):
                return False
        return True

    def check_const_inputs(self, node_name):
        """Check the node has the const input

        Args:
            node_name (string): node name

        Returns:
            bool: True if the node has the const input else False
        """
        if node_name not in self.graph_info:
            return False
        node_op = self.graph_info[node_name].node.op
        if node_op == "Placeholder" or node_op == "Const":
            return False
        if node_op not in self.supported_op_type:
            return False
        constant_flag = True
        for input_name in self.graph_info[node_name].node.input:
            input_name = GraphRewriterHelper.node_name_from_input(input_name)
            input_node = self.graph_info[input_name].node
            constant_flag &= input_node.op == "Const" and not input_node.input
        return constant_flag

    def do_transformation(self):
        """fold all the sequences only consist of const and self.supported_op_type

        Args:
          input_graph_def (graphdef): graphdef object

        Returns:
           [graphdef]: optimized graph
        """

        while not self.check_all_folded():
            for node_name, _ in self.graph_info.copy().items():
                if self.check_const_inputs(node_name):
                    fold_value = self._fold_value(node_name)
                    fold_type = tf.as_dtype(fold_value.dtype)
                    new_constant_node = GraphRewriterHelper.create_constant_node(
                        node_name + "_const", fold_value, fold_type)
                    self.graph_analyzer.replace_constant_graph_with_constant_node(
                        new_constant_node, node_name)

        output_graph_def = self.graph_analyzer.dump_graph()

        return output_graph_def
