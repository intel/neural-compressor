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

import os
import logging
from pathlib import Path
from neural_compressor.utils.utility import LazyImport
from neural_compressor.model.base_model import BaseModel

onnx = LazyImport('onnx')
ort = LazyImport("onnxruntime")

logger = logging.getLogger()


def find_by_name(item_name, item_list):
    '''
    Helper function to find item by name in a list.
        parameter item_name: name of the item.
        parameter item_list: list of items.
        return: item if found. None otherwise.
    '''
    items = [item for item in item_list if item.name == item_name]
    return items[0] if len(items) > 0 else None


class ONNXModel(BaseModel):
    def __init__(self, model, **kwargs):
        self._model = model if not isinstance(model, str) else onnx.load(model)
        self.node_name_counter = {}
        self._output_name_to_node = {}
        self._input_name_to_nodes = {}
        self._get_input_name_to_nodes(self._model.graph.node)
        self._get_output_name_to_node(self._model.graph.node)
        self._graph_info = {}
        self._get_graph_info()
        self._q_config = None

    def framework(self):
        return 'onnxruntime'

    @property
    def q_config(self):
        return self._q_config

    @q_config.setter
    def q_config(self, q_config):
        self._q_config = q_config

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        self._graph_info = {}
        self._get_graph_info()
        self._output_name_to_node = {}
        self._input_name_to_nodes = {}
        self._get_input_name_to_nodes(self._model.graph.node)
        self._get_output_name_to_node(self._model.graph.node)

    def input(self):
        return [i.name for i in self._model.graph.input]

    def output(self):
        return [i.name for i in self._model.graph.output]

    def update(self):
        self._graph_info = {}
        self._get_graph_info()
        self._output_name_to_node = {}
        self._input_name_to_nodes = {}
        self._get_input_name_to_nodes(self._model.graph.node)
        self._get_output_name_to_node(self._model.graph.node)

    @property
    def graph_info(self):
        return self._graph_info

    def _get_graph_info(self):
        for node in self._model.graph.node:
            self.graph_info.update({node.name: node.op_type})

    def save(self, root):
        if os.path.split(root)[0] != '' and not os.path.exists(os.path.split(root)[0]):
            raise ValueError('"root" directory does not exists.')
        onnx.save(self._model, root)

    def nodes(self):
        return self._model.graph.node

    def initializer(self):
        return self._model.graph.initializer

    def graph(self):
        return self._model.graph

    def ir_version(self):
        return self._model.ir_version

    def opset_import(self):
        return self._model.opset_import

    def remove_node(self, node):
        if node in self._model.graph.node:
            self._model.graph.node.remove(node)

    def remove_nodes(self, nodes_to_remove):
        for node in nodes_to_remove:
            self.remove_node(node)

    def add_node(self, node):
        self._model.graph.node.extend([node])

    def add_nodes(self, nodes_to_add):
        self._model.graph.node.extend(nodes_to_add)

    def add_initializer(self, tensor):
        if find_by_name(tensor.name, self._model.graph.initializer) is None:
            self._model.graph.initializer.extend([tensor])

    def get_initializer(self, name):
        for tensor in self._model.graph.initializer:
            if tensor.name == name:
                return tensor
        return None

    def remove_initializer(self, tensor):
        if tensor in self._model.graph.initializer:
            self._model.graph.initializer.remove(tensor)

    def remove_initializers(self, init_to_remove):
        for initializer in init_to_remove:
            self.remove_initializer(initializer)

    def set_initializer(self, tensor, array):
        old_tensor = self.get_initializer(tensor)
        self.remove_initializer(old_tensor)
        dims = old_tensor.dims
        data_type = old_tensor.data_type
        new_tensor = onnx.helper.make_tensor(tensor, data_type, dims, array.flatten().tolist())
        self.add_initializer(new_tensor)
    
    @property
    def input_name_to_nodes(self):
        return self._input_name_to_nodes
    
    def _get_input_name_to_nodes(self, nodes):
        for node in nodes:
            attrs = [attr for attr in node.attribute if attr.type == onnx.AttributeProto.GRAPH \
                or attr.type == onnx.AttributeProto.GRAPHS]
            if len(attrs) > 0:
                for attr in attrs:
                    self._get_input_name_to_nodes(attr.g.node)
            for input_name in node.input:
                if input_name not in self._input_name_to_nodes:
                    self._input_name_to_nodes[input_name] = [node]
                else:
                    self._input_name_to_nodes[input_name].append(node)

    @property
    def output_name_to_node(self):
        return self._output_name_to_node

    def _get_output_name_to_node(self, nodes):
        for node in nodes:
            attrs = [attr for attr in node.attribute if attr.type == onnx.AttributeProto.GRAPH \
                or attr.type == onnx.AttributeProto.GRAPHS]
            if len(attrs) > 0:
                for attr in attrs:
                    self._get_output_name_to_node(attr.g.node)
            for output_name in node.output:
                self._output_name_to_node[output_name] = node

    def get_children(self, node, input_name_to_nodes=None):
        if input_name_to_nodes is None:
            input_name_to_nodes = self._input_name_to_nodes

        children = []
        for output in node.output:
            if output in input_name_to_nodes:
                for child in input_name_to_nodes[output]:
                    children.append(child)
        return children

    def get_parents(self, node, output_name_to_node=None):
        if output_name_to_node is None:
            output_name_to_node = self._output_name_to_node

        parents = []
        for input in node.input:
            if input in output_name_to_node:
                parents.append(output_name_to_node[input])
        return parents

    def get_parent(self, node, idx, output_name_to_node=None):
        if output_name_to_node is None:
            output_name_to_node = self._output_name_to_node

        if len(node.input) <= idx:
            return None

        input = node.input[idx]
        if input not in output_name_to_node:
            return None

        return output_name_to_node[input]

    def find_node_by_name(self, node_name, new_nodes_list, graph):
        '''
        Find out if a node exists in a graph or a node is in the
        new set of nodes created during quantization. Return the node found.
        '''
        graph_nodes_list = list(graph.node)  #deep copy
        graph_nodes_list.extend(new_nodes_list)
        node = find_by_name(node_name, graph_nodes_list)
        return node

    def find_nodes_by_initializer(self, graph, initializer):
        '''
        Find all nodes with given initializer as an input.
        '''
        nodes = []
        for node in graph.node:
            for node_input in node.input:
                if node_input == initializer.name:
                    nodes.append(node)
        return nodes

    def get_scale_zero(self, tensor):
        ''' help function to get scale and zero_point '''
        if not tensor.endswith('_quantized'):
            logger.debug("Find {} in the quantized graph is not quantized.".format(tensor))
            return None, None
        input_name_to_nodes = self._input_name_to_nodes
        node = input_name_to_nodes[tensor][0]
        scale = "_".join(tensor.split('_')[:-1] + ['scale'])
        scale_tensor = self.get_initializer(scale)
        zo = "_".join(tensor.split('_')[:-1] + ['zero_point'])
        zo_tensor = self.get_initializer(zo)

        #TODO check if scale_tensor and zero_point is needed
        # for bias of qlinearconv, scale and zero_point is not needed
        if node.op_type == 'QLinearConv' and tensor == node.input[-1]:
            pass
        else:
            assert scale_tensor, 'missing scale for tensor {}'.format(tensor)
            assert zo_tensor, 'missing zero point for tensor {}'.format(tensor)
        return scale_tensor, zo_tensor

    def save_model_to_file(self, output_path, use_external_data_format=False):
        '''
        Save model to external data, which is needed for model size > 2GB
        '''
        if use_external_data_format:
            onnx.external_data_helper.convert_model_to_external_data(self._model,
                                                    all_tensors_to_one_file=True,
                                                    location=Path(output_path).name + ".data")
        onnx.save_model(self._model, output_path)

    @staticmethod
    def replace_node_input(node, old_input_name, new_input_name):
        assert isinstance(old_input_name, str) and isinstance(new_input_name, str)
        for j in range(len(node.input)):
            if node.input[j] == old_input_name:
                node.input[j] = new_input_name

    def replace_input_of_all_nodes(self, old_input_name, new_input_name,
        white_optype=[], black_optype=[]):
        if len(white_optype) > 0:
            for node in self.model.graph.node:
                if node.op_type in white_optype:
                    ONNXModel.replace_node_input(node, old_input_name, new_input_name)
        else:
            for node in self.model.graph.node:
                if node.op_type not in black_optype:
                    ONNXModel.replace_node_input(node, old_input_name, new_input_name)


    @staticmethod
    def replace_node_output(node, old_output_name, new_output_name):
        assert isinstance(old_output_name, str) and isinstance(new_output_name, str)
        for j in range(len(node.output)):
            if node.output[j] == old_output_name:
                node.output[j] = new_output_name

    def replace_output_of_all_nodes(self, old_output_name, new_output_name, 
        white_optype=[], black_optype=[]):
        if len(white_optype) > 0:
            for node in self.model.graph.node:
                if node.op_type in white_optype:
                    ONNXModel.replace_node_output(node, old_output_name, new_output_name)
        else:
            for node in self.model.graph.node:
                if node.op_type not in black_optype:
                    ONNXModel.replace_node_output(node, old_output_name, new_output_name)

    def remove_unused_constant(self):
        unused_nodes = []
        nodes = self.nodes()
        for node in nodes:
            if node.op_type == "Constant" and node.output[0] not in self._model.graph.output \
                and node.output[0] not in self._input_name_to_nodes:
                unused_nodes.append(node)
            elif node.op_type == 'QuantizeLinear' and len(self.get_children(node)) == 1 and \
                self.get_children(node)[0].op_type == 'DequantizeLinear' and \
                node.input[0] not in self._output_name_to_node and \
                self.get_children(node)[0].output[0] not in self._input_name_to_nodes:
                unused_nodes.append(node)
                unused_nodes.extend(self.get_children(node))
        self.remove_nodes(unused_nodes)

        ununsed_weights = []
        for w in self._model.graph.initializer:
            if w.name not in self._input_name_to_nodes and w.name not in self._model.graph.output:
                ununsed_weights.append(w)
                # Remove from graph.input
                for graph_input in self.graph().input:
                    if graph_input.name == w.name:
                        self.graph().input.remove(graph_input)

        self.remove_initializers(ununsed_weights)
