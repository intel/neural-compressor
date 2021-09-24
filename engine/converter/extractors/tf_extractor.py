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

from tensorflow.python.framework import tensor_util
import tensorflow as tf
from neural_compressor.utils import logger
import numpy as np
from ..graph.graph import Graph
from ..ops.op import OPERATORS
from ..tf_utils import graph_node_names_details
from ..graph_utils import names_from_input


class TensorflowExtractor(object):
    """
    Decorate the node in model.graph_def, and the new node has the attributes like input_tensors
    and output_tensors, these tensors record the source/dest op name. All of these nodes
    (in a list) will compose a graph, which is Graph class, as the return object.
    Args:
        model: neural_compressor TensorflowBaseModel
    Return:
        Graph: Graph class, the new graph object

    """
    @classmethod
    def __call__(self, model):

        nodes = model.graph_def.node
        graph_nodes_dict = graph_node_names_details(nodes)
        logger.info('Start to extarct tensorflow model ops...')
        new_graph = Graph()
        new_graph.framework = 'tensorflow'
        for node in nodes:
            # ignore the Const nodes, they have no source inputs
            if node.op == 'Const':
                continue
            else:
                op_type = node.op
                if op_type in OPERATORS.keys():
                    new_node = OPERATORS[op_type]()
                    new_node.extract('tensorflow', node, model, graph_nodes_dict)
                    new_graph.insert_nodes(len(new_graph.nodes), [new_node])
                else:
                    raise ValueError('the {} operation does not support now...'.format(op_type))

        return new_graph
