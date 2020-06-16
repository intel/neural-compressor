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

import logging

from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import gfile


class GraphTransformBase(object):
    def __init__(self, input_pb):
        """
        Basic class for graph transformation.
        Parameters:
             input_pb: the input graphdef or pb file.
        """
        if isinstance(input_pb, graph_pb2.GraphDef):
            self.input_graph = input_pb
        else:
            try:
                with gfile.Open(input_pb, 'rb') as f:
                    self.input_graph.ParseFromString(f.read())
            except Exception as e:
                logging.error("Failed to read input pb: {} due to {}".format(input_pb, str(e)))

        self.node_mapping = {}
        self.node_name_list = []

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
                logging.warning('Duplicate node name {}'.format(node.name))

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
