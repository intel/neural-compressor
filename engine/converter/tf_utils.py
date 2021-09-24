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

from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import tensor_util
import numpy as np
import re
from collections import namedtuple, OrderedDict
from neural_compressor.utils import logger
from .ops.tensor import Tensor
from . import graph_utils as util


def create_tf_node(op, name, inputs):
    """Create a nodedef object
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


def graph_node_names_details(nodes):
    """Parse the graph nodes ans get the graph_nodes_dict.
    Be used for Grpah class with cerating a new graph.
    The node_name is the key, node in value is for getting the Const
    tensor value and the input_tensor source op; outputs in value is for
    output_tensor dest op.
    Args:
        nodes (tendorflow graph_def.node): NodeDef list
    Returns:
        node_names_details: the graph node info dict

    """
    # nodes = model.graph_def.node
    # some node may have several output edges
    # use tensor represents edge in graph
    node_details = namedtuple('node_details', ['node', 'outputs'])
    node_names_details = {}
    for node in nodes:
        node_name = node.name
        each_node = node_details(node=node, outputs=[])
        if node_name not in node_names_details:
            node_names_details[node_name] = each_node
    for name, details in node_names_details.items():
        # node.input represents input tensors name, not nodes names
        # in many times, they are same string list
        inputs = []
        for each_input in details.node.input:
            each_input = util.names_from_input(each_input)[0]
            if each_input not in inputs:
                inputs.append(each_input)
        for each_i in inputs:
            node_names_details[each_i].outputs.append(name)
    return node_names_details


def get_tensor_dest_op(node_name, tensor_name, nodes_dict):
    """get the tensor dest op name
       Args:
        node_name: string, the source node name of tensor
        tensor_name: string, with ':0' or something like it
        nodes_dict: dict returned by the graph_node_names_details function
       Returns:
        dest_op_names: list, store the tensor's dest op names
    """
    dest_op_names = []
    post_node_names = nodes_dict[node_name].outputs
    for n_name in post_node_names:
        for input_name in nodes_dict[n_name].node.input:
            if util.names_from_input(input_name)[1] == tensor_name:
                dest_op_names.append(n_name)
            else:
                continue
    return dest_op_names


def tf_extract_operator(node, model, nodes_dict):
    """decorate the operator in tensorflow
    Args:
        node: NodeDef
        model: neural_compressor TensorflowBaseModel
        nodes_dict: dict, return value from graph_node_names_details
        tf_dtypes: dict, for get the dtype string

    Returns:
        op_type: node op type
        input_tensors: Tensor list, contains the node input tensors info
        output_tensors: Tensor list, contains the node output tensor info
    """
    op_type = node.op
    input_tensors = []
    output_tensors = []

    input_names = node.input
    for i, n in enumerate(input_names):
        # for Identity Constant
        if n.endswith('/read'):
            input_names[i] = n[:-5]
        else:
            continue

    """ input_tensors
    each input_tensor has its own soure op, but same dest op
    so both have single string
    """
    # name list
    input_tensor_names = input_names
    for input_tensor_name in input_tensor_names:
        input_node_name, input_tensor_name = util.names_from_input(input_tensor_name)
        pre_node = nodes_dict[input_node_name].node
        if pre_node.op == 'Const':
            data = tensor_util.MakeNdarray(pre_node.attr['value'].tensor)
            dtype = util.get_data_dtype(data)
            shape = list(data.shape) if data.shape != () else [1]
            input_tensor = Tensor(name=input_tensor_name,
                                  source_op=[],
                                  dest_op=[node.name],
                                  shape=shape,
                                  data=data,
                                  dtype=dtype
                                  )
            input_tensors.append(input_tensor)
        else:
            input_tensor = Tensor(name=input_tensor_name,
                                  source_op=[input_node_name],
                                  dest_op=[node.name],
                                  shape=None,
                                  data=None,
                                  dtype=None
                                  )
            input_tensors.append(input_tensor)

    """ output_tensors
    Almost every op generate one tensor in deep learning, so we just give one tensor in 
    output_tensors (list). However, this tensor maybe delivered to several nodes, so the 
    dest_op should have several strings.
    """
    if node.op not in MULTI_OUTPUT_OP.keys():
        out_num = 1
    else:
        if node.op == 'Unpack':
            out_num = node.attr['num'].i
        elif node.op == 'IteratorGetNext':
            out_num = len(node.attr['output_types'].list.type)
        else:
            out_num = MULTI_OUTPUT_OP[node.op]

    for i in range(out_num):
        output_tensor_name = node.name + ':' + str(i)
        try:
            output_tensor = model.graph.get_tensor_by_name(output_tensor_name)
        except BaseException:
            return op_type, input_tensors, []

        dest_op = get_tensor_dest_op(node.name, output_tensor_name, nodes_dict)
        output_tensor = Tensor(name=output_tensor_name,
                               source_op=[node.name],
                               dest_op=dest_op,
                               shape=None,
                               data=None,
                               dtype=None
                               )

        output_tensors.append(output_tensor)

    return op_type, input_tensors, output_tensors


TF_DTYPE_ID = {3: 'int32',
               1: 'fp32',
               9: 'int64',
               7: 'string', }

MULTI_OUTPUT_OP = {'Unpack': -1,
                   'IteratorGetNext': -1,
                   }
