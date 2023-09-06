#
#  -*- coding: utf-8 -*-
#
#  Copyright (c) 2021 Intel Corporation
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
"""GraphTransform Base Class."""

from __future__ import absolute_import, division, print_function

import logging
import re

from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import gfile

logger = logging.getLogger("neural_compressor")


class GraphTransformBase(object):
    """GraphTransform Base Class."""

    def __init__(self, input_pb):
        """Basic class for graph transformation.

        Args:
             input_pb: the input graphdef or pb file.
        """
        if isinstance(input_pb, graph_pb2.GraphDef):
            self.input_graph = input_pb
        else:
            try:
                with gfile.Open(input_pb, "rb") as f:
                    self.input_graph.ParseFromString(f.read())
            except Exception as e:
                logger.error("Fail to read input pb from {} due to {}.".format(input_pb, str(e)))

        self.node_mapping = {}
        self.node_name_list = []
        self.output_node_map = {}
        self.generate_input_map()

    def parse_input_pb(self):
        """Parse the input pbdef to get the node name and node mapping.

        Returns:
            the dict that key is node name while the value is nodeDef.
        """
        for node in self.input_graph.node:
            self.node_name_list.append(node.name)

            if node.name not in self.node_mapping:
                self.node_mapping[node.name] = node
            else:
                logger.warning("Duplicated node name {}.".format(node.name))

    def generate_input_map(self):
        """Generate the input map."""
        self.input_node_map = {}
        for node in self.input_graph.node:
            node_name = self.node_name_from_input(node.name)
            if node_name not in self.input_node_map:
                self.input_node_map[node_name] = node
            else:
                raise ValueError("Duplicate node names detected for ", node.name)

    def node_name_from_input(self, node_name):
        """Get the original node name from input string.

        Args:
            node_name: input node's name in string

        Returns:
            node's name
        """
        if node_name.startswith("^"):  # pragma: no cover
            node_name = node_name[1:]
        m = re.search(r"(.*):\d+$", node_name)
        if m:
            node_name = m.group(1)
        return node_name

    def get_node_name_from_input(self, node_name):
        """Get the original node name from input string.

        Args:
            node_name: input node's name in string

        Returns:
            node's name
        """
        node_names = node_name.split(":")
        return node_names[0]

    def do_transformation(self):
        """Virtual Interface.

        Each transformation should implement it.
        """
        pass
