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
import copy
import logging
from pathlib import Path
from neural_compressor.model.base_model import BaseModel

logger = logging.getLogger()

class EngineModel(BaseModel):
    def __init__(self, model, **kwargs):
        self._graph = None
        self._model = model
        self.q_config = None

    def framework(self):
        return 'engine'
    
    def _load_graph(self, model):
        if os.path.isdir(model):
            from engine.converter.graph import Graph
            file_list = os.listdir(model)
            assert len(file_list) == 2
            for file_name in file_list:
                file_ext= os.path.splitext(file_name)
                front, ext = file_ext
                if ext == ".yaml":
                    yaml_path = os.path.join(model, file_name)
                elif ext == ".bin":
                    bin_path = os.path.join(model, file_name)
                else:
                    logger.error("Please Input yaml and bin for engine.")
            graph = Graph()
            graph.graph_init(yaml_path, bin_path)
            logger.warn("When input yaml and bin, it can not use func in converter.")
        else:
            from engine.converter.loaders.loader import Loader
            from engine.converter.extractors.extractor import Extractor
            from engine.converter.sub_graph.subgraph_matcher import SubGraphMatcher
            loader = Loader()
            extractor = Extractor()
            sub_graph_matcher = SubGraphMatcher()
            graph = loader(model)
            graph = extractor(graph)
            graph = sub_graph_matcher(graph)
        return graph

    @property
    def graph(self):
        if not self._graph:
            self._graph = self._load_graph(self.model)
        return self._graph

    @graph.setter
    def graph(self, graph):
        self._graph = graph

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def save(self, output_dir=None):
        self.graph.save(output_dir)

    @property
    def nodes(self):
        return self.graph.nodes

    @nodes.setter
    def nodes(self, new_nodes):
        self.graph.nodes = new_nodes

    def insert_nodes(self, index, nodes):
        self.graph.insert_nodes(index, nodes)

    def remove_nodes(self, node_names):
        self.graph.remove_nodes(node_names)
    
    def get_node_id(self, node_name):
        return self.graph.get_node_id(node_name)

    def get_node_by_name(self, node_name):
        return self.graph.get_node_by_name(node_name)

    def rename_node(self, old_name, new_name):
        self.graph.rename_node(old_name, new_name)

    def get_pre_node_names(self, node_name):
        return self.graph.get_pre_node_names(node_name)

    def get_next_node_names(self, node_name):
        return self.graph.get_next_node_names(node_name)

    def modify_node_connections(self, node, mode='insert'):
        return self.graph.modify_node_connections(node, mode)

    def dump_tensor(self, tensor_list = []):
        return self.graph.dump_tensor(tensor_list)

    def engine_init(self, net_info={}, weight_data=b""):
        return self.graph.engine_init(net_info={}, weight_data=b"")

    def inference(self, input_data):
        return self.graph.inference(input_data)

    def change_node_input_tensors(self, node_name, index, tensor=None, mode='modify'):
        return self.graph.change_node_input_tensors(node_name, index, tensor, mode)
