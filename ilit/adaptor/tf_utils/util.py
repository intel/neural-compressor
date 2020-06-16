#
#  -*- coding: utf-8 -*-
#
#  Copyright (c) 2019 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import os

from google.protobuf import text_format

import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.platform import gfile


def read_graph(in_graph, in_graph_is_binary=True):
    """Reads input graph file as GraphDef.

    :param in_graph: input graph file.
    :param in_graph_is_binary: whether input graph is binary, default True.
    :return: input graphDef.
    """
    if not gfile.Exists(in_graph):
        raise ValueError('Input graph pb file %s does not exist.' % in_graph)

    input_graph_def = graph_pb2.GraphDef()
    mode = "rb" if in_graph_is_binary else "r"
    with gfile.Open(in_graph, mode) as f:
        data = f.read()
        if in_graph_is_binary:
            input_graph_def.ParseFromString(data)
        else:
            text_format.Merge(data, input_graph_def)

    return input_graph_def


def write_graph(out_graph_def, out_graph_file):
    """Write output graphDef to file.

    :param out_graph_def: output graphDef.
    :param out_graph_file: path to output graph file.
    :return: None.
    """
    if not isinstance(out_graph_def, tf.compat.v1.GraphDef):
        raise ValueError('out_graph_def is not instance of TensorFlow GraphDef.')
    if out_graph_file and not os.path.exists(os.path.dirname(out_graph_file)):
        raise ValueError('"output_graph" directory does not exists.')
    f = gfile.GFile(out_graph_file, 'wb')
    f.write(out_graph_def.SerializeToString())


def split_shared_inputs(in_graph, ops=[]):
    """
    Split shared inputs(like weights and bias) of ops list.
    :param in_graph: input graph file.
    :param ops: ops list to processing.
    :return: path to ouput graph file.
    """
    if not ops:
        return in_graph

    input_graph_def = read_graph(in_graph)

    # map of node_name - node
    node_map = {}
    for node in input_graph_def.node:
        if node.name not in node_map.keys():
            node_map[node.name] = node

    output_graph_def = graph_pb2.GraphDef()
    # map of input_name - op_name
    input_map = {}
    for node_name in node_map.keys():
        node = node_map[node_name]
        if node.op in ops:
            for input_idx, input_node_name in enumerate(node.input):
                if node_map[input_node_name].op == 'Const':
                    # is shared and current node is not the first one sharing the input
                    if input_node_name in input_map.keys():
                        input_map[input_node_name].append(node.name)
                        new_input_node = node_def_pb2.NodeDef()
                        new_input_node.CopyFrom(node_map[input_node_name])
                        new_input_node.name = input_node_name + '_' + str(len(input_map[input_node_name]))
                        node.input[input_idx] = new_input_node.name
                        output_graph_def.node.extend([new_input_node])
                    else:
                        input_map[input_node_name] = [node.name]
        output_graph_def.node.extend([node])
    rewrite_graph = os.path.join(os.path.dirname(in_graph), 'frozen_inference_graph_rewrite.pb')
    write_graph(output_graph_def, rewrite_graph)
    return rewrite_graph
