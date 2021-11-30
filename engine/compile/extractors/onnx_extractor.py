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


from neural_compressor.utils import logger
from ..graph.graph import Graph
from ..ops.op import OPERATORS
from ..onnx_utils import graph_node_names_details
from ..graph_utils import names_from_input


class ONNXExtractor(object):
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
        # onnx_consts = model.initializer()
        graph_nodes_dict = graph_node_names_details(model)
        logger.info('Start to extarct onnx model ops...')
        new_graph = Graph()
        new_graph.framework = 'onnxruntime'
        for graph_input in model.graph().input:
            op_type = 'ONNXINPUT'
            new_node = OPERATORS[op_type]()
            new_node.extract('onnxruntime', graph_input, model, graph_nodes_dict)
            new_graph.insert_nodes(len(new_graph.nodes), [new_node])
        for node in model.nodes():
            op_type = node.op_type
            if op_type == 'Constant':
                continue
            else:
                if op_type in OPERATORS.keys():
                    new_node = OPERATORS[op_type]()
                    new_node.extract('onnxruntime', node, model, graph_nodes_dict)
                    new_graph.insert_nodes(len(new_graph.nodes), [new_node])
                    if op_type == 'Loop':
                        for inner_node in node.attribute[0].g.node:
                            op_type = inner_node.op_type
                            if op_type == 'Constant':
                                continue
                            else:
                                import engine.compile.graph_utils as util
                                input_tensor_names = inner_node.input
                                for input_tensor_name in input_tensor_names:
                                    origin_tensor_name, input_tensor_name = \
                                        util.names_from_input(input_tensor_name)
                                    if origin_tensor_name in graph_nodes_dict and \
                                        hasattr(graph_nodes_dict[origin_tensor_name], 'node'):
                                        pre_node = graph_nodes_dict[origin_tensor_name].node
                                        data = None
                                        graph_node = new_graph.get_node_by_name(node.name)
                                        has_tensor = True
                                        for tensor in graph_node.input_tensors:
                                            if origin_tensor_name + ':0' == tensor.name: 
                                                has_tensor = False
                                        if pre_node in model.initializer() and has_tensor:
                                            from onnx.numpy_helper import to_array
                                            data = to_array(pre_node)
                                            from engine.compile.ops.tensor import Tensor
                                            shape = list(data.shape) if data.shape != () else [1]
                                            dtype = util.get_data_dtype(data)
                                            loop_tensor = Tensor(name=origin_tensor_name,
                                                                 source_op=[],
                                                                 dest_op=[node.name],
                                                                 shape=shape,
                                                                 data=data,
                                                                 dtype=dtype
                                                                )
                                            new_graph.change_node_input_tensors(node.name, 0, \
                                                                     loop_tensor, mode='insert')    

                else:
                    raise ValueError('the {} operation does not support now...'.format(op_type))

        return new_graph
