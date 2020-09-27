#
#  -*- coding: utf-8 -*-
#

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import tf_logging
from ..graph_base import GraphRewriterBase
from ..graph_util import TFGraphAnalyzer, TFGraphRewriterHelper


class GraphFoldConstantOptimizer(GraphRewriterBase):
    supported_op_type = ["Add", "AddV2", "Const", "Identity", "Mul", "Rsqrt", "Sub"]

    def __init__(self, model=None):
        super(GraphFoldConstantOptimizer, self).__init__(model)
        graph_analyzer = TFGraphAnalyzer()
        graph_analyzer.graph = self.model

        self.graph_info = graph_analyzer.parse_graph()

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
                fold_value = np.array([1.])
                for index, input in enumerate(end_node.input):
                    # broadcast if needed
                    input_value = self._fold_value(input)
                    if can_broadcast(fold_value, input_value):
                        fold_value = fold_value * input_value
                    else:
                        raise ValueError("input {} of node {} can't be broadcast".format(
                            input.name, end_node.name))
                return fold_value
            elif end_node.op == "Add" or end_node.op == "AddV2":
                fold_value = np.array([0.])
                for index, input in enumerate(end_node.input):
                    # broadcast if needed
                    input_value = self._fold_value(input)
                    if can_broadcast(fold_value, input_value):
                        fold_value = fold_value + input_value
                    else:
                        raise ValueError("input {} of node {} can't be broadcast".format(
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
                    if can_broadcast(fold_value, input_value):
                        fold_value = fold_value + (-1)**index * input_value
                    else:
                        raise ValueError("input {} of node {} can't be broadcast".format(
                            input.name, end_node.name))
                return fold_value
            else:
                tf_logging.info(
                    "Currently fold-constant only support limited ops {} but face {}".format(
                        self.supported_ops, end_node.op))
        else:
            return np.float32(TFGraphRewriterHelper.values_from_const(end_node))

    def check_all_folded(self):
        for node_name, _ in self.graph_info.items():
            if self.check_const_inputs(node_name):
                return False
        return True

    def check_const_inputs(self, node_name):
        node_op = self.graph_info[node_name].node.op
        if node_op == "Placeholder" or node_op == "Const":
            return False
        if node_op not in self.supported_op_type:
            return False
        constant_flag = True
        for input_name in self.graph_info[node_name].node.input:
            input_name = TFGraphRewriterHelper.node_name_from_input(input_name)
            input_node = self.graph_info[input_name].node
            constant_flag &= input_node.op == "Const"
        return constant_flag

    def do_transformation(self):
        """fold all the sequences only consist of const and self.supported_ops

        Args:
          input_graph_def (graphdef): graphdef object

        Returns:
           [graphdef]: optimized graph
        """

        graph_analyzer = TFGraphAnalyzer()

        while not self.check_all_folded():
            for node_name, _ in self.graph_info.copy().items():
                if self.check_const_inputs(node_name):
                    outputs = self.graph_info[node_name].outputs
                    fold_value = self._fold_value(node_name)
                    fold_type = tf.as_dtype(np.float32(fold_value).dtype)
                    new_constant_node = TFGraphRewriterHelper.create_constant_node(
                        node_name + "_const", fold_value, fold_type)
                    graph_analyzer.replace_constant_graph_with_constant_node(
                        new_constant_node, node_name, outputs)

        output_graph_def = graph_analyzer.dump_graph()

        return output_graph_def
