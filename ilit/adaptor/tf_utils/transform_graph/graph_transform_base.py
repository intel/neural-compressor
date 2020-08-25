#
#  -*- coding: utf-8 -*-
#
#  Copyright (c) 2019 Intel Corporation
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
from tensorflow.python.platform import gfile
from tensorflow.python.framework import tensor_util
import logging
import re

class GraphTransformBase(object):
    def __init__(self, input_pb):
        """
        Basic class for graph transformation.
        Parameters:
             input_pb: the input graphdef or pb file.
        """
        self.logger = logging.getLogger()

        if isinstance(input_pb, graph_pb2.GraphDef):
            self.input_graph = input_pb
        else:
            try:
                with gfile.Open(input_pb, 'rb') as f:
                    self.input_graph.ParseFromString(f.read())
            except Exception as e:
                self.logger.error("Failed to read input pb: {} due to {}".format(
                    input_pb, str(e)))

        self.node_mapping = {}
        self.node_name_list = []
        self.output_node_map = {}
        self.generate_input_map()

    def parse_input_pb(self):
        """
        Parse the input pbdef to get the node name and node mapping.
        Returns:
            the dict that key is node name while the value is nodeDef.
        """
        for node in self.input_graph.node:
            self.node_name_list.append(node.name)

            if node.name not in self.node_mapping:
                self.node_mapping[node.name] = node
            else:
                self.logger.warning('Duplicate node name {}'.format(node.name))

    def get_input_nodes(self):
        self.input_nodes = []
        for node in self.input_graph.node:
            if node.input == []:
                if node.op == "Placeholder" or node.op == "Const":
                    self.input_nodes.append(node)
                else:
                    raise ValueError(
                        "graph input should only be Placeholder or Const, found {} in {}".format(
                            node, node.op))

    def generate_input_map(self):
        self.input_node_map = {}
        for node in self.input_graph.node:
            node_name = self.node_name_from_input(node.name)
            if node_name not in self.input_node_map:
                self.input_node_map[node_name] = node
            else:
                raise ValueError("Duplicate node names detected for ",
                                 node.name)

    def node_name_from_input(self, node_name):
        """Get the original node name from input string.
        Parameters:
            node_name: input node's name in string
        Returns:
            node's name
        """
        if node_name.startswith("^"):
            node_name = node_name[1:]
        m = re.search(r"(.*):\d+$", node_name)
        if m:
            node_name = m.group(1)
        return node_name

    def node_from_map(self, node_map, name):
        """Pulls a node def from a dictionary for a given name.

        Args:
          node_map: Dictionary containing an entry indexed by name for every node.
          name: Identifies the node we want to find.

        Returns:
          NodeDef of the node with the given name.

        Raises:
          ValueError: If the node isn't present in the dictionary.
        """
        stripped_name = self.node_name_from_input(name)
        if stripped_name not in node_map:
            raise ValueError("No node named '%s' found in map." % name)
        return node_map[stripped_name]

    def values_from_const(self, node_def):
        """Extracts the values from a const NodeDef as a numpy ndarray.

        Args:
          node_def: Const NodeDef that has the values we want to access.

        Returns:
          Numpy ndarray containing the values.

        Raises:
          ValueError: If the node isn't a Const.
        """
        if node_def.op != "Const":
            raise ValueError(
                "Node named '%s' should be a Const op for values_from_const." %
                node_def.name)
        input_tensor = node_def.attr["value"].tensor
        tensor_value = tensor_util.MakeNdarray(input_tensor)
        return tensor_value

    def get_node_name_from_input(self, node_name):
        """
        Get the original node name from input string.
        Parameters:
            node_name: input node's name in string
        Returns:
            node's name
        """
        node_names = node_name.split(':')
        return node_names[0]

    def do_transformation(self):
        """
        Virtual Interface. Each transformation should implement it.
        """
        pass
