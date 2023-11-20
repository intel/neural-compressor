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
"""Tensorflow Graph Utils Helper Classes."""

import copy
import logging
import re
from collections import namedtuple

import numpy as np
from tensorflow.core.framework import attr_value_pb2, graph_pb2, node_def_pb2
from tensorflow.python.framework import tensor_util

from neural_compressor.utils.utility import singleton

logger = logging.getLogger("neural_compressor")


@singleton
class GraphAnalyzer:
    """Tensorflow Graph Analyzer class which implemented under singleton mode.

    This class provides the following API:
    * Analyze the graph
    * Analyze the input/output node names of the specified graph
    """

    # TODO add the positive input flag
    node_details = namedtuple("node_details", ["node", "outputs"])

    def __init__(self, extend_engine=None):
        """Initialization.

        Args:
            extend_engine: extended engine, for future extension API。
        """
        self._graph = None
        self.extend_engine = extend_engine

    @property
    def graph(self):
        """Getter of the _graph object.

        Returns:
            graph: current graphdef object
        """
        return self._graph

    @graph.setter
    def graph(self, new_graph):
        """Update the internal graph value.

        Args:
            new_graph (graphdef object): new model object
        """
        self._graph = new_graph

    def _has_positive_input(self, start_node):
        """Check the start_node if has positive input."""
        op_type = start_node.op
        if op_type in ("Relu", "Relu6") or op_type.find("AndRelu") != -1:
            return True
        elif op_type.startswith("Quantized") and not op_type.endswith("AndRelu"):
            return False
        elif op_type in ("Concat", "Add", "AddV2", "AddN"):
            for each_input in start_node.input:
                has_relu = self._has_positive_input(
                    self.node_name_details[GraphRewriterHelper.node_name_from_input(each_input)].node
                )
                if not has_relu:
                    return False
            return True
        elif op_type in (
            "Conv3D",
            "Conv2D",
            "DepthwiseConv2D",
            "QuantizeV2",
            "DepthwiseConv2dNative",
            "MaxPool",
            "MaxPool3D",
            "Requantize",
            "AvgPool",
            "Pad",
            "CropAndResize",
            "Dequantize",
            "Mean",
            "MatMul",
            "FusedBatchNormV3",
            "_MklFusedInstanceNorm",
        ):
            return self._has_positive_input(
                self.node_name_details[GraphRewriterHelper.node_name_from_input(start_node.input[0])].node
            )
        else:
            return False

    def has_positive_input(self, node_name):
        """Check the specified node has positive input data or not.

        Args:
            node_name (string): node name

        Returns:
            bool: return True if the node has the positive input data,
                return False if the node has the negative input data.
        """
        return self._has_positive_input(self.node_name_details[node_name].node)

    def get_graph_input_output(self):
        """Get the graphdef input/output node names.

        Sometimes, the configuration doesn't specifies the input/output names of the graph,
        but tensorflow need to know them clearly to run the graph.We implement this function has the similar
        feature like summarize_graph.py which writtern by Google.

        Returns:
            tuple: (inputs' name list, outputs'name list)
        """
        input_node_names = []
        output_node_names = []
        unlikely_output_types = [
            "Const",
            "HostConst",
            "Assign",
            "NoOp",
            "Parameter",
            "Assert",
            "save",
            "global_step",
            "read",
            "switch",
            "cond",
            "train",
            "init_ops",
            "[A-Za-z]+Dataset",
        ]
        unlikely_input_types = [
            "FIFOQueueV2",
            "QueueDequeueV2",
            "QueueDequeueUpToV2",
            "OneShotIterator",
            "IteratorGetNext",
            "IteratorV2",
        ]
        exclude_input_names = []
        extra_input_names = []

        for _, i in self.node_name_details.items():
            for exclude_input_name in exclude_input_names:
                if exclude_input_name == i.node.name:
                    if i.node.op in unlikely_input_types:
                        exclude_input_names += i.outputs
                    else:
                        extra_input_names.append(i.node.name)
            if i.node.op in ["Const", "HostConst", "Variable", "VariableV2"]:
                continue
            if not i.node.input and not i.outputs:
                logger.debug("Skip isolated node {}.".format(i.node.name))
            elif i.node.op == "Placeholder":
                input_node_names.append(i.node.name)
            elif not i.node.input:
                if i.node.op not in unlikely_input_types:
                    input_node_names.append(i.node.name)
                else:
                    exclude_input_names += i.outputs
            elif (
                not i.outputs
                and i.node.op not in unlikely_output_types
                and not re.match(unlikely_output_types[-1], i.node.op)
            ):
                output_node_names.append(i.node.name)
            else:
                pass

        if len(input_node_names) == 0 and len(extra_input_names) != 0:
            for extra_input_name in extra_input_names:
                input_node_names.append(extra_input_name)

        logger.warning(
            "Found possible input node names: {}, output node names: {}.".format(input_node_names, output_node_names)
        )

        return (input_node_names, output_node_names)

    def query_fusion_pattern_nodes(self, patterns=None):
        """Public interface for query the nodes aggregation status.

        Args:
            patterns (string list): Please check the _search_patterns definition.

        Returns:
            [string list]: The matched node names which saved as the string list.
        """
        if self.extend_engine:
            # Todo keep this for future extension API
            pass
        else:
            return self._search_patterns(patterns)

    def _search_patterns(self, input_pattern):
        """Search user specified patterns on internal grpah structure.

        Args:
            input_pattern (list): The element of the pattern list could be string/list/tuple.
            string or list means the specified types are mandatory while tuple stands for optional.
            e.g:
            ['Conv2D', ['BiasAdd'], ("Add", "AddN"), ["Relu","Relu6"]] it equals to below patterns:
            Conv2D + BiasAdd + Add + Relu
            Conv2D + BiasAdd + AddN + Relu
            Conv2D + BiasAdd + Add + Relu6
            Conv2D + BiasAdd + AddN + Relu6
            Conv2D + BiasAdd + Relu
            Conv2D + BiasAdd + Relu6

        Return: [string list]. Each matched pattern composed of matched node name and we put the
                    match node op as the last element of each pair.
                    e.g
                    [
                        ['resnet_model/conv2d_4/Conv2D',
                        'resnet_model/batch_normalization_4/FusedBatchNorm',
                        'resnet_model/add',
                        'resnet_model/Relu_3',
                        ['Conv2D', 'BiasAdd', 'Add', 'Relu']],
                        ['resnet_model/conv2d_7/Conv2D',
                        'resnet_model/batch_normalization_7/FusedBatchNorm',
                        'resnet_model/add_1',
                        'resnet_model/Relu_6',
                        ['Conv2D', 'BiasAdd', 'AddN', 'Relu6']]
                    ]
        """

        def _validate_input(data, criteria):
            if isinstance(criteria, str) and data == criteria:
                return True

            if isinstance(criteria, (list, tuple)) and data in criteria:
                return True

            return False

        def _compare_list(list_a, list_b):
            """Check list a is a subset of list b.

            e.g, list a is ['a', 'b', 'c'] while list b is ['a', 'b', 'c', 'd'],
            then list a is subset of list b.

            Args:
                list_a ([Any]): list A
                list_b ([Any]): list B

            Returns:
                [bool]: list a is a subset of list b or not.
            """
            assert isinstance(list_a, list)
            assert isinstance(list_b, list)
            is_subset = True

            for index, value in enumerate(list_a):
                is_subset &= value == list_b[index]

            return is_subset

        def _dfs(op_names, op_types, graph_info, node, pattern):
            if pattern == []:
                return
            start_index = 0
            end_index = len(pattern) - 1
            matched_flag = False
            while start_index <= end_index:
                matched_flag = _validate_input(node.op, pattern[end_index])

                if not matched_flag and isinstance(pattern[end_index], tuple):
                    end_index -= 1
                    continue

                if matched_flag:
                    op_names.append(node.name)
                    op_types.append(node.op)
                    break

                return

            if start_index == end_index:
                if matched_flag:
                    matched_res = copy.deepcopy(op_names)
                    matched_res.reverse()
                    op_types_copy = copy.deepcopy(op_types)
                    op_types_copy.reverse()
                    matched_res.append(op_types_copy)
                    if matched_res not in output_result:
                        output_result.append(matched_res)

                    op_names.pop()
                    op_types.pop()
                return

            for index, value in enumerate(node.input):
                cur_node = graph_info[GraphRewriterHelper.node_name_from_input(value)].node
                _dfs(op_names, op_types, graph_info, cur_node, pattern[:end_index])
                if index == len(node.input) - 1:
                    op_names.pop()
                    op_types.pop()

        output_result = []

        for _, v in self.node_name_details.items():
            start_index = len(input_pattern) - 1
            while start_index >= 0:
                find_first_match = _validate_input(v.node.op, input_pattern[start_index])
                if find_first_match:
                    break

                if isinstance(input_pattern[start_index], tuple):
                    start_index -= 1
                    continue

                start_index = -2

            if start_index < 0:
                continue

            visited_op_name = []
            visited_op_types = []

            _dfs(visited_op_name, visited_op_types, self.node_name_details, v.node, input_pattern)

        sorted_output = sorted(output_result, key=lambda i: i[-1])

        useless_match_list = []
        for index, value in enumerate(sorted_output):
            if index == len(sorted_output) - 1:
                break

            next_matched_op_names = sorted_output[index + 1][:-1]
            if len(value[:-1]) < len(next_matched_op_names) and _compare_list(value[:-1], next_matched_op_names):
                useless_match_list.append(value)

        for i in useless_match_list:
            sorted_output.remove(i)

        longest_match = {}
        final_output = []
        for i in sorted_output:
            key = i[0]
            if key not in longest_match:
                longest_match[key] = i[-1]
                continue

            if len(longest_match[key]) < len(i[-1]):
                longest_match[key] = i[-1]

        for i in sorted_output:
            if i[0] in longest_match and i[-1] == longest_match[i[0]]:
                final_output.append(i)

        return final_output

    def remove_node_with_single_input_output(self, node_name):
        """Remove node with one input and rebuild internal graph data structure.

        Args:
            node_name (string): node name

        Returns:
            [bool]: True if remove the node without exception,
                    False if failed to remove it.
        """
        if node_name not in self.node_name_details:
            logger.debug("The {} is not a valid node name.".format(node_name))
            return False

        non_const_node_count = len(
            [
                GraphRewriterHelper.node_name_from_input(i)
                for i in self.node_name_details[node_name].node.input
                if self.node_name_details[GraphRewriterHelper.node_name_from_input(i)].node.op != "Const"
            ]
        )

        if non_const_node_count > 1:
            logger.debug("The target node {} has more than one input.".format(node_name))
            return False

        try:
            top_node_name = GraphRewriterHelper.node_name_from_input(self.node_name_details[node_name].node.input[0])

            for bottom_node_name in self.node_name_details[node_name].outputs:
                update_output_name = [
                    bottom_node_name if i == node_name else i for i in self.node_name_details[top_node_name].outputs
                ]
                self.node_name_details[top_node_name]._replace(outputs=update_output_name)

                update_input_name = [
                    self.node_name_details[node_name].node.input[0] if i == node_name else i
                    for i in self.node_name_details[bottom_node_name].node.input
                ]

                if self.node_name_details[bottom_node_name].node.input:
                    self.node_name_details[bottom_node_name].node.ClearField("input")
                    self.node_name_details[bottom_node_name].node.input.extend(update_input_name)

        except Exception as e:
            logger.debug("Fail to remove node {} due to {}.".format(node_name, str(e)))
            return False
        else:
            return self.remove_node(node_name)

    def remove_node(self, node_name):
        """Remove the user specified node by its name.

        Args:
            node_name (string): node name string.

        Returns:
            [bool]: True if remove the node without exception.
                    False if failed to remove it.
        """
        if node_name not in self.node_name_details:
            logger.debug("The {} is not a valid node name.".format(node_name))
            return False
        try:
            self.node_name_details.pop(node_name)
        except Exception as e:
            logger.info("Fail to remove {} due to {}.".format(node_name, str(e)))
            return False
        else:
            logger.debug("{} has been removed.".format(node_name))
            return True

    def replace_const_node(self, new_const_node, target_node, old_constant_node_name, replace_all=True):
        """Replace the specified const node with another one.

        Args:
            new_const_node (NodeDef): node name string.
            target_node (list): the string list that contains name of node that
                                need to be replaced const node.
            old_constant_node_name (string): the outdated const node name.
            replace_all (bool): replace the specified node name once or not.
        """
        new_const_node_name = new_const_node.name

        self.node_name_details[new_const_node_name] = self.node_details(node=new_const_node, outputs=target_node)

        for sub_node in target_node:
            if sub_node not in self.node_name_details:
                continue
            for index, each_node_name in enumerate(self.node_name_details[sub_node].node.input):
                if each_node_name + ":0" == old_constant_node_name or each_node_name == old_constant_node_name:
                    new_input_name = (
                        self.node_name_details[sub_node].node.input[:index]
                        + [new_const_node_name]
                        + self.node_name_details[sub_node].node.input[index + 1 :]
                    )
                    self.node_name_details[sub_node].node.ClearField("input")
                    self.node_name_details[sub_node].node.input.extend(new_input_name)
                    if old_constant_node_name in self.node_name_details:
                        self.node_name_details[old_constant_node_name].outputs.remove(sub_node)
                        if len(self.node_name_details[old_constant_node_name].outputs) == 0:
                            self.remove_node(old_constant_node_name)
                    if not replace_all:
                        break

    def replace_constant_graph_with_constant_node(self, new_node, old_end_node_name):
        """Remove sub-graph with a const node.

        Args:
            new_node (nodedef): the constant node
            old_end_node_name (string):  the sub-graph end node which will be updated by new node

        Returns:
            [bool]: True if remove the node without exception.
                    False if failed to remove it.
        """
        new_node_name = new_node.name

        if new_node.op != "Const":
            logger.warning("The input of replace_with_constant_node must be a constant node.")
            return False
        try:
            inputs = self.node_name_details[old_end_node_name].node.input
            inputs = [GraphRewriterHelper.node_name_from_input(i) for i in inputs]
            for input_name in inputs:
                if self.node_name_details[input_name].node.op != "Const":
                    logger.warning("The subgraph replaces must be constant.")
                    return False
                elif len(self.node_name_details[input_name].outputs) == 1:
                    self.node_name_details.pop(input_name)
            output_node_name = self.node_name_details[old_end_node_name].outputs
            self.replace_node(new_node, old_end_node_name, output_node_name)
            self.node_name_details[new_node_name].node.ClearField("input")
        except Exception as e:
            logger.info("Fail to replace {} due to {}.".format(old_end_node_name, str(e)))
            return False
        else:
            return True

    def replace_single_node(
        self, new_node, old_output_node_names, old_output_name, old_input_node_names, old_input_name
    ):
        """Insert one node into the graph.

        Args:
            new_node (nodedef): new nodedef object
            old_output_node_names (string list):the node names that would be the top node of new
                                                node.
            old_output_name (string list): the names that need to be updated with new node name
            old_input_node_names (string list): the node names that would be the bottom node of new
                                                node.
            old_input_name (string list): the names that need to be updated with new node name
        """
        new_node_name = new_node.name
        for i in old_output_node_names:
            while old_output_name in self.node_name_details[i].outputs:
                self.node_name_details[i].outputs.remove(old_output_name)
            self.node_name_details[i].outputs.append(new_node_name)

        self.node_name_details[new_node_name] = self.node_details(node=new_node, outputs=old_input_node_names)

        for each_input_node_name in old_input_node_names:
            for index, each_node_name in enumerate(self.node_name_details[each_input_node_name].node.input):
                if self.node_name_details[each_input_node_name].node.input and (each_node_name) == old_input_name:
                    new_input_name = (
                        self.node_name_details[each_input_node_name].node.input[:index]
                        + [new_node_name]
                        + self.node_name_details[each_input_node_name].node.input[index + 1 :]
                    )
                    self.node_name_details[each_input_node_name].node.ClearField("input")
                    self.node_name_details[each_input_node_name].node.input.extend(new_input_name)

    def replace_node(self, new_node, old_node_name, output_nodes_name):
        """Replace the node into the internal data structure node_name_details.

        Args:
            new_node (nodedef): the nodedef object.
            old_node_name (string): the parent node of input node.
            output_nodes_name (string list): output node names list
        """
        new_node_name = new_node.name
        self.node_name_details[new_node_name] = self.node_details(node=new_node, outputs=output_nodes_name)
        old_node = self.node_name_details[old_node_name].node
        for input_node_name in old_node.input:
            if input_node_name in self.node_name_details:
                self.node_name_details[input_node_name].outputs.remove(old_node_name)
                self.node_name_details[input_node_name].outputs.append(new_node_name)

        for node_name in output_nodes_name:
            for index, each_node_name in enumerate(self.node_name_details[node_name].node.input):
                if (
                    self.node_name_details[node_name].node.input
                    and GraphRewriterHelper.node_name_from_input(each_node_name) == old_node_name
                ):
                    new_input_name = (
                        self.node_name_details[node_name].node.input[:index]
                        + [new_node_name]
                        + self.node_name_details[node_name].node.input[index + 1 :]
                    )
                    self.node_name_details[node_name].node.ClearField("input")
                    self.node_name_details[node_name].node.input.extend(new_input_name)
        self.remove_node(old_node_name)

    def add_node(self, new_node, start_node_name, end_node_names):
        """Add the node into the internal data structure node_name_details.

        Args:
            new_node (nodedef): the nodedef object.
            start_node_name (string): the parent node of input node.
            end_node_names (string list): output node names list
        """
        new_node_name = new_node.name

        if new_node_name in self.node_name_details:
            logger.debug("Remove the existed node {} from internal data structure.".format((new_node_name)))
            self.node_name_details.pop(new_node_name)

        self.node_name_details[new_node_name] = self.node_details(node=new_node, outputs=end_node_names)

        for end_node_name in end_node_names:
            # Update start node's output info
            if end_node_name not in self.node_name_details:
                continue
            if (
                start_node_name
                and end_node_name
                in self.node_name_details[GraphRewriterHelper.node_name_from_input(start_node_name)].outputs
            ):
                self.node_name_details[GraphRewriterHelper.node_name_from_input(start_node_name)].outputs.remove(
                    end_node_name
                )

            # reset output node's input
            for index, each_node_name in enumerate(self.node_name_details[end_node_name].node.input):
                if each_node_name == start_node_name:
                    new_input_name = (
                        self.node_name_details[end_node_name].node.input[:index]
                        + [new_node_name]
                        + self.node_name_details[end_node_name].node.input[index + 1 :]
                    )
                    self.node_name_details[end_node_name].node.ClearField("input")
                    self.node_name_details[end_node_name].node.input.extend(new_input_name)

        # add the inserted node into the start node's output.
        if start_node_name:
            self.node_name_details[GraphRewriterHelper.node_name_from_input(start_node_name)].outputs.append(
                new_node_name
            )

    def dump_graph(self):
        """Dump the current model's graphdef.

        Returns:
            [graphdef]: A graphdef object
        """
        output_graph_def = graph_pb2.GraphDef()
        for _, v in self.node_name_details.items():
            output_graph_def.node.extend([v.node])

        return output_graph_def

    def get_frame_info(self):
        """Get the frame info of the model.

        Returns:
            [parent_frame_details]: OrderedDict frame info of the graph nodes.
        """
        from collections import OrderedDict

        self.parent_frame_details = OrderedDict()
        input_node_names, _ = self.get_graph_input_output()

        traverse_list = copy.deepcopy(input_node_names)
        visited = []

        while traverse_list:
            node_name = traverse_list.pop(0)
            node_details = self.node_name_details[node_name]

            if node_details.node.name in visited:
                continue

            for output in node_details.outputs:
                traverse_list.append(output)

                inputs = node_details.node.input
                if not inputs:
                    self.parent_frame_details[node_details.node.name] = None
                if self.node_name_details[output].node.op == "Enter":
                    self.parent_frame_details[output] = self.node_name_details[output].node
                elif self.node_name_details[output].node.op == "Exit":
                    self.parent_frame_details[output] = None
                else:
                    if output in self.parent_frame_details and self.parent_frame_details[output]:
                        if (
                            node_details.node.name in self.parent_frame_details
                            and self.parent_frame_details[node_details.node.name]
                        ):
                            assert (
                                self.parent_frame_details[output].attr["frame_name"]
                                == self.parent_frame_details[node_details.node.name].attr["frame_name"]
                            )
                    else:
                        if node_details.node.name in self.parent_frame_details:
                            self.parent_frame_details[output] = self.parent_frame_details[node_details.node.name]

            visited.append(node_details.node.name)
        return self.parent_frame_details

    def parse_graph(self, input_graph_def=None):
        """Analyze the input graphdef and return the list contains each node's input/outputnode names.

        Args:
            input_graph_def ([graphdef]): graphdef object

        Returns:
            [list]: A list contains each node's inputs/outputs info.
        """
        if not input_graph_def:
            input_graph_def = self._graph

        self.node_name_details = {}

        for node in input_graph_def.node:
            node_name = GraphRewriterHelper.node_name_from_input(node.name)

            each_node = self.node_details(node=node, outputs=[])

            if node_name not in self.node_name_details:
                self.node_name_details[node_name] = each_node

        for node_name, node_details in self.node_name_details.items():
            # update the upper node's output information.
            for each_input in node_details.node.input:
                self.node_name_details[GraphRewriterHelper.node_name_from_input(each_input)].outputs.append(node_name)

        return self.node_name_details


class GraphRewriterHelper:
    """Encapsulates the graph operation into one class."""

    node_name_cache = {}
    node_name_port_cache = {}

    @staticmethod
    def compare_node_attr(node_a, node_b):
        """Compare two node has identical attributes or not.

        Args:
            node_a (nodedef): Input node.
            node_b (nodedef): Another node to be compared.

        Returns:
            [bool]: True if two node have the identical attributes.
        """
        if len(node_a.input) > 1:
            return False

        if node_a.input != node_b.input:
            return False

        if node_a.op != node_b.op:
            return False

        if len(node_a.attr) != len(node_b.attr):
            return False

        node_a_attr = sorted(list(node_a.attr))
        node_b_attr = sorted(list(node_b.attr))

        if node_a_attr != node_b_attr:
            return False

        for attr_name in node_a_attr:
            if node_a.attr[attr_name] != node_b.attr[attr_name]:
                return False

        return True

    @staticmethod
    def create_node(op, name, inputs):
        """Create a nodedef object.

        Args:
            op (string): op type
            name (string): op name
            inputs (string list): op's inputs name

        Returns:
            nodedef: the created nodedef object
        """
        new_node = node_def_pb2.NodeDef()
        new_node.op = op
        new_node.name = name
        for input_name in inputs:
            new_node.input.extend([input_name])
        return new_node

    @staticmethod
    def create_constant_node(name, value, dtype, shape=None, device="cpu"):
        """Create constant node.

        Args:
            name (string): op name
            value (np.array): input data
            dtype (datatype): data type of the input value
            shape (int list, optional): the value's shape. Defaults to None.
            device (str, optional): the device type, it may be the 'cpu' or 'gpu'.
                                    Defaults to 'cpu'.

        Returns:
            [type]: [description]
        """
        node = GraphRewriterHelper.create_node("Const" if device == "cpu" else "HostConst", name, [])
        GraphRewriterHelper.set_attr_dtype(node, "dtype", dtype)
        GraphRewriterHelper.set_attr_tensor(node, "value", value, dtype, shape)
        return node

    @staticmethod
    def set_attr_dtype(node, key, value):
        """Set the attribute data type."""
        node.attr[key].CopyFrom(attr_value_pb2.AttrValue(type=value.as_datatype_enum))

    @staticmethod
    def set_attr_tensor(node, key, value, dtype, shape=None):
        """Set the tensor value to specified attribute field.

        Args:
            node (nodedef): the target nodedef object
            key (string): attribute name
            value (np.array): the content
            dtype (dtypes): data type
            shape (int list, optional): the input tensor's shape. Defaults to None.
        """
        node.attr[key].CopyFrom(
            attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(value, dtype=dtype, shape=shape))
        )

    @staticmethod
    def set_attr_type_list(node, key, value):
        """Set the node's attr which data type is int list."""
        list_value = attr_value_pb2.AttrValue.ListValue(type=value)
        node.attr[key].CopyFrom(attr_value_pb2.AttrValue(list=list_value))

    @staticmethod
    def set_attr_string_list(node, key, value):
        """Set the node's attr which data type is int list."""
        list_value = attr_value_pb2.AttrValue.ListValue(s=value)
        node.attr[key].CopyFrom(attr_value_pb2.AttrValue(list=list_value))

    @staticmethod
    def set_attr_string(node, key, value):
        """Set the node's attr which data type is string."""
        node.attr[key].CopyFrom(attr_value_pb2.AttrValue(s=value))

    @staticmethod
    def set_attr_int_list(node, key, value):
        """Set the node's attr which data type is int list."""
        list_value = attr_value_pb2.AttrValue.ListValue(i=value)
        node.attr[key].CopyFrom(attr_value_pb2.AttrValue(list=list_value))

    @staticmethod
    def set_attr_int(node, key, value):
        """Set the node's attr which data type is int."""
        node.attr[key].CopyFrom(attr_value_pb2.AttrValue(i=value))

    @staticmethod
    def set_attr_float(node, key, value):
        """Set the node's attr which data type is float."""
        node.attr[key].CopyFrom(attr_value_pb2.AttrValue(f=value))

    @staticmethod
    def set_attr_bool(node, key, value):
        """Set the node's attr which data type is bool."""
        node.attr[key].CopyFrom(attr_value_pb2.AttrValue(b=value))

    @staticmethod
    def node_name_from_input(node_name):
        """Static method that get the valid node name from input name.

        Args:
            node_name (string): node name defined in the input field.

        Returns:
            string: node's name
        """
        if node_name not in GraphRewriterHelper.node_name_cache:
            key = node_name
            if node_name.startswith("^"):
                node_name = node_name[1:]
            m = re.search(r"(.*):\d+$", node_name)
            if m:
                node_name = m.group(1)
            GraphRewriterHelper.node_name_cache[key] = node_name
            return node_name

        return GraphRewriterHelper.node_name_cache[node_name]

    @staticmethod
    def values_from_const(node_def):
        """Extracts the values from a const NodeDef as a numpy ndarray.

        Args:
          node_def: Const NodeDef that has the values we want to access.

        Returns:
          Numpy ndarray containing the values.

        Raises:
          ValueError: If the node isn't a Const.
        """
        assert node_def.op == "Const", "Node named '%s' should be a Const op." % node_def.name

        input_tensor = node_def.attr["value"].tensor
        tensor_value = tensor_util.MakeNdarray(input_tensor)
        return tensor_value

    @staticmethod
    def generate_int32_bias_for_conv(
        bias_tensor,
        channel_size,
        max_input,
        min_input,
        max_filter_tensor,
        min_filter_tensor,
        activation_range,
        weights_range=127.0,
    ):
        """Static method that generate int32 bias for conv op.

        Args:
            bias_tensor: bias node tensor.
            channel_size: channel size.
            max_input: max activation input value.
            min_input: min activation input value.
            max_filter_tensor: max weight input tensor.
            min_filter_tensor: min weight input tensor.
            activation_range: activation range value.
            weights_range: weight range value.

        Returns:
            int32_bias: int32 bias
        """
        bias_length = bias_tensor.shape[0]
        scales = []
        if len(max_filter_tensor) > 1:
            for i in range(channel_size):
                scales.append(
                    activation_range
                    * weights_range
                    / (max(abs(max_input), abs(min_input)) * max(abs(max_filter_tensor[i]), abs(min_filter_tensor[i])))
                )
        else:
            for i in range(channel_size):
                scales.append(
                    activation_range
                    * weights_range
                    / (max(abs(max_input), abs(min_input)) * max(abs(max_filter_tensor[0]), abs(min_filter_tensor[0])))
                )
        int32_bias = []
        if channel_size > 1:
            for i in range(bias_length):
                int32_bias.append((int)(np.around(bias_tensor[i] * scales[i])))
        else:
            for i in range(bias_length):
                int32_bias.append((int)(np.around(bias_tensor[i] * scales[0])))

        return int32_bias

    @staticmethod
    def generate_int32_bias_for_matmul(
        bias_tensor,
        weights_tensor,
        input_range,
        max_input,
        min_input,
        max_filter_value,
        min_filter_value,
    ):
        """Static method that generate int32 bias for matmul op.

        Args:
            bias_tensor: bias node tensor.
            weights_tensor: weights tensor.
            input_range: activation range value.
            max_input: max activation input value.
            min_input: min activation input value.
            max_filter_tensor: max weight input tensor.
            min_filter_tensor: min weight input tensor.

        Returns:
            int32_bias: int32 bias
        """
        bias_scale = 255.0 * 127.0 / (input_range * max(abs(max_filter_value), abs(min_filter_value)))
        relative_scale = 255 * min_input / (max_input - min_input)
        int32_bias = []
        for bias_index, value in enumerate(np.sum(np.array(weights_tensor, dtype=np.int32), axis=0, dtype=np.int32)):
            if bias_index >= bias_tensor.size:
                continue
            int32_bias.append(int(np.around(bias_tensor[bias_index] * bias_scale + value * relative_scale)))

        return int32_bias

    @staticmethod
    def generate_int32_bias_for_matmul_per_channel(
        bias_tensor,
        weights_tensor,
        max_input,
        min_input,
        max_filter_tensor,
        min_filter_tensor,
    ):  # pragma: no cover
        """Static method that generate per-channel int32 bias for matmul op.

        Args:
            bias_tensor: bias node tensor.
            weights_tensor: weights tensor.
            max_input: max activation input value.
            min_input: min activation input value.
            max_filter_tensor: max weight input tensor.
            min_filter_tensor: min weight input tensor.

        Returns:
            int32_bias: int32 bias
        """
        channel_size = bias_tensor.shape[0]
        activation_range = 255.0
        weights_range = 127.0
        scales = []
        relative_scale = 255 * min_input / (max_input - min_input)
        for i in range(channel_size):
            scales.append(
                activation_range
                * weights_range
                / ((max_input - min_input) * max(abs(max_filter_tensor[i]), abs(min_filter_tensor[i])))
            )
        int32_bias = []
        for i in range(channel_size):
            value = np.sum(np.array(weights_tensor), axis=0, dtype=np.int32)[i]
            int32_bias.append((int)(np.around(value * relative_scale + bias_tensor[i] * scales[i])))

        return int32_bias

    @staticmethod
    def gen_valid_sampling_log(log_path):
        """Generate the valid sampling log.

        Args:
          log_path: the valid sampling log file path.

        Returns:
          the sampling min max value.
        """

        def gen_per_iter(data):
            res = []
            requant_tmp = []
            for i in data:
                if i.find("__print__;__requant_") == -1:
                    res.append(i)
                else:
                    requant_tmp.append(i)
            sorted_requant = sorted(requant_tmp)
            odd_list = sorted_requant[::2]
            even_list = sorted_requant[1::2]
            for index, value in enumerate(even_list):
                min_value = min(0, float(value.split(":")[1][1:-1]))
                max_value = float(odd_list[index].split(":")[1][1:-1])
                max_value = max_value if max_value > min_value else min_value + 1e-05
                mixed_str = value.split(":")[0] + "_max:[" + str(min_value) + "][" + str(max_value) + "]"

                res.append(mixed_str)
            return res

        def separate(line):
            """This function is to separate the strings.

            Example:
                ';slice__print__;__max:[1];slice__print__;__min:[-1]' -->
                [';slice__print__;__max:[1]', ';slice__print__;__min:[-1]']
            """
            separated_lines = []
            for subline in line.split("];"):
                if not subline.startswith(";"):
                    subline = ";" + subline
                if not subline.endswith("]"):
                    subline += "]"
                separated_lines.append(subline)
            return separated_lines

        with open(log_path) as f:
            valid_data = []
            for i in f.readlines():
                if not i.startswith(";"):
                    continue
                line = i.strip()
                if line.find("];") != 0:
                    separated_lines = separate(line)
                    valid_data += separated_lines
                else:
                    valid_data.append(line)

        first_line = valid_data[0].rsplit(":")[0]

        iterations = 0
        for i in valid_data:
            if i.startswith(first_line):
                iterations += 1

        step = int(len(valid_data) / iterations)
        if step % 2 == 1:
            step -= 1
            iterations = int(len(valid_data) / step) + int(len(valid_data) % step > 0)

        final_res = []

        for i in range(iterations):
            final_res.extend(gen_per_iter(valid_data[int(i * step) : int(step * (i + 1))]))
            if i + 1 == iterations and int(step * (i + 1)) < len(valid_data):
                final_res.extend(gen_per_iter(valid_data[int(step * (i + 1)) : len(valid_data)]))

        return final_res

    @staticmethod
    def analysis_rnn_model(graph_def, bf16_ops=[], fp32_ops=[]):
        """Match the RNN and dynamic RNN patterns."""
        g = GraphAnalyzer()
        g.graph = graph_def
        graph_info = g.parse_graph()
        rnn_pattern = [["TensorArrayV3"], ["Enter"], ["TensorArrayReadV3"], ["MatMul"], ["BiasAdd"]]
        target_nodes = g.query_fusion_pattern_nodes(rnn_pattern)
        res = {}
        for i in target_nodes:
            if i[-3] not in bf16_ops and i[-3] not in fp32_ops:
                res[(i[-3], i[-2])] = graph_info[i[1]].node.attr["frame_name"].s.decode()

        dynamic_rnn_pattern = [["Enter"], ["MatMul"], ["BiasAdd"]]
        target_nodes = g.query_fusion_pattern_nodes(dynamic_rnn_pattern)
        for i in target_nodes:
            if i[-3] not in bf16_ops and i[-3] not in fp32_ops:
                res[(i[1], i[2])] = graph_info[i[0]].node.attr["frame_name"].s.decode()

        return res
