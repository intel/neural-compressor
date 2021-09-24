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

import re
from collections import OrderedDict
from neural_compressor.utils import logger
import numpy as np
import yaml
import os


class Graph(object):
    def __init__(self):
        self._nodes = []
        self._node_id = {}
        self._engine = None

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, new_nodes):
        self._nodes = new_nodes

    def insert_nodes(self, index, nodes):
        idx = index
        for node in nodes:
            node = self.modify_node_connections(node, mode='insert')
            self._nodes.insert(idx, node)
            self._node_id[node.name] = idx
            for i in range(idx + 1, len(self._nodes)):
                self._node_id[self._nodes[i].name] += 1
            idx += 1
        self._engine = None

    def remove_nodes(self, node_names):
        for node_name in node_names:
            if node_name not in self._node_id.keys():
                continue
            node = self.get_node_by_name(node_name)
            _ = self.modify_node_connections(node, mode='remove')
            index = self.get_node_id(node_name)
            for i in range(index + 1, len(self._nodes)):
                self._node_id[self._nodes[i].name] -= 1
            self._nodes.pop(index)
            self._node_id.pop(node_name)
        self._engine = None

    def get_node_id(self, node_name):
        try:
            index = self._node_id[node_name]
            return index
        except BaseException:
            raise ValueError(
                'There is no node named {}, please check the input name.'.format(node_name))

    def get_node_by_name(self, node_name):
        index = self.get_node_id(node_name)
        if index is not None:
            return self._nodes[index]
        else:
            return None

    def rename_node(self, old_name, new_name):
        index = self.get_node_id(old_name)
        for i in range(len(self._nodes[index].input_tensors)):
            self._nodes[index].input_tensors[i].dest_op = [new_name]
            for pre_node_name in self._nodes[index].input_tensors[i].source_op:
                tensor_idx = self.get_tensor_idx(pre_node_name, 
                                            self._nodes[index].input_tensors[i].name)
                pre_node_idx = self._node_id[pre_node_name]
                self._nodes[pre_node_idx].output_tensors[tensor_idx].dest_op.remove(old_name)
                self._nodes[pre_node_idx].output_tensors[tensor_idx].dest_op.append(new_name)
        for i in range(len(self._nodes[index].output_tensors)):
            self._nodes[index].output_tensors[i].source_op = [new_name]
            for next_node_name in self._nodes[index].output_tensors[i].dest_op:
                tensor_idx = self.get_tensor_idx(next_node_name, 
                                self._nodes[index].output_tensors[i].name, from_output=False)
                next_node_idx = self._node_id[next_node_name]
                self._nodes[next_node_idx].input_tensors[tensor_idx].source_op = [new_name]

        self._nodes[index].name = new_name
        self._node_id.pop(old_name)
        self._node_id[new_name] = index
        self._engine = None

    def change_node_input_tensors(self, node_name, index, tensor=None, mode='modify'):
        assert mode in ['insert', 'remove', 'modify'], 'Wrong mode'
        node = self.get_node_by_name(node_name)
        index = index if index != -1 else len(node.input_tensors) - 1
        node_index = self.get_node_id(node_name)
        source_node_idx = None
        tensor_idx = None
        if mode == 'remove':
            tensor = node.input_tensors[index]
        assert tensor is not None
        tensor.dest_op = [node_name]
        if tensor.source_op != []:
            source_node_idx = self.get_node_id(tensor.source_op[0])
            tensor_idx = self.get_tensor_idx(tensor.source_op[0], tensor.name, from_output=True)

        if mode == 'insert':
            if source_node_idx is not None:
                if node_name not in \
                self._nodes[source_node_idx].output_tensors[tensor_idx].dest_op:
                    self._nodes[source_node_idx].output_tensors[tensor_idx].dest_op.append(
                        node_name)
            self._nodes[node_index].input_tensors.insert(index, tensor)
        elif mode == 'remove':
            if source_node_idx is not None:
                self._nodes[source_node_idx].output_tensors[tensor_idx].dest_op.remove(node_name)
            self._nodes[node_index].input_tensors.pop(index)

        else:
            self.change_node_input_tensors(node_name, index, mode='remove')
            self.change_node_input_tensors(node_name, index, tensor=tensor, mode='insert')
        self._engine = None

    def change_node_output_tensors(self, node_name, index, tensor=None, mode='modify'):
        assert mode in ['insert', 'remove', 'modify'], 'Wrong mode'
        node = self.get_node_by_name(node_name)
        index = index if index != -1 else len(node.output_tensors) - 1
        node_index = self.get_node_id(node_name)
        if mode == 'remove':
            tensor = node.output_tensors[index]
        assert tensor is not None
        tensor.source_op = [node_name]

        if mode == 'insert':
            self._nodes[node_index].output_tensors.insert(index, tensor)
        elif mode == 'remove':
            self._nodes[node_index].output_tensors.pop(index)
        else:
            self._nodes[node_index].output_tensors[index] = tensor
        self._engine = None

    def get_pre_node_names(self, node_name):
        pre_node_names = []
        node = self.get_node_by_name(node_name)
        for input_tensor in node.input_tensors:
            if input_tensor.source_op != []:
                pre_node_names.extend(input_tensor.source_op)

        return pre_node_names

    def get_next_node_names(self, node_name):
        next_node_names = []
        node = self.get_node_by_name(node_name)
        for output_tensor in node.output_tensors:
            if output_tensor.dest_op != []:
                next_node_names.extend(output_tensor.dest_op)

        return next_node_names

    def get_tensor_idx(self, node_name, tensor_name, from_output=True):
        target_node = self.get_node_by_name(node_name)
        tensor_idx = -1
        if from_output:
            target_tensors = target_node.output_tensors
        else:
            target_tensors = target_node.input_tensors
        for j in range(len(target_tensors)):
            if target_tensors[j].name == tensor_name:
                tensor_idx = j
                break
            else:
                continue
        # assert tensor_idx != -1, 'Graph does not has tensor {}, '\
        #'please check it.'.format(tensor_name)

        return tensor_idx

    def modify_node_connections(self, node, mode='insert'):
        assert mode in ['insert', 'remove'], 'Wrong mode {}'.format(mode)
        # modify the input_tensors' source_op
        for i in range(len(node.input_tensors)):
            node.input_tensors[i].dest_op = [node.name]
            t = node.input_tensors[i]
            if t.source_op != [] and t.source_op[0] in self._node_id.keys():
                source_node_idx = self.get_node_id(t.source_op[0])
                source_node = self._nodes[source_node_idx]
                tensor_idx = self.get_tensor_idx(source_node.name, t.name)
                if mode == 'insert':
                    if node.name not in \
                    self._nodes[source_node_idx].output_tensors[tensor_idx].dest_op:
                        self._nodes[source_node_idx].output_tensors[tensor_idx].dest_op.append(
                            node.name)
                if mode == 'remove':
                    self._nodes[source_node_idx].output_tensors[tensor_idx].dest_op.remove(
                        node.name)
            # skip the const tensor and the node has been removed
            else:
                continue

        # modify the output_tensors' dest_op
        if mode == 'insert':
            for i in range(len(node.output_tensors)):
                node.output_tensors[i].source_op = [node.name]
                t = node.output_tensors[i]
                for dest_op_name in node.output_tensors[i].dest_op:
                    if dest_op_name in self._node_id.keys():
                        dest_node_idx = self.get_node_id(dest_op_name)
                        tensor_idx = self.get_tensor_idx(dest_op_name, t.name, from_output=False)
                        if tensor_idx != -1:
                            self._nodes[dest_node_idx].input_tensors[tensor_idx].source_op = [
                                node.name]
        self._engine = None

        return node

    # get the weight_bytes to bin file
    @property
    def weight_data(self):
        consts_info = OrderedDict()
        weight_bytes = bytearray()
        non_consts_len = 0
        for t in self._nodes[0].output_tensors:
            assert self._nodes[0].op_type=='Input', 'The graph must have input data'
            if t.source_op==[] and isinstance(t.data, np.ndarray):
                break
            else:
                non_consts_len += 1
        self._nodes[0].output_tensors = self._nodes[0].output_tensors[:non_consts_len]
        for i in range(len(self._nodes)):
            for j in range(len(self._nodes[i].input_tensors)):
                t = self._nodes[i].input_tensors[j]
                if t.source_op==[] and isinstance(t.data, np.ndarray):
                    data = t.data
                    start = len(weight_bytes)
                    data_bytes = data.tobytes()
                    weight_bytes.extend(data_bytes)
                    offset = len(data_bytes)
                    self._nodes[i].input_tensors[j].location = [start, offset]
                    self._nodes[0].output_tensors.append(self._nodes[i].input_tensors[j])
        weight_bytes = bytes(weight_bytes)
        return weight_bytes

    # get the network config dict to yaml file
    @property
    def net_config(self):
        net_info = OrderedDict()
        net_info['model'] = OrderedDict()
        net_info['model']['name'] = 'model'
        net_info['model']['operator'] = OrderedDict()
        for node in self._nodes:
            net_info['model']['operator'][node.name] = node.config

        return net_info

    def dump_tensor(self, tensor_list=[]):
        weight_data = self.weight_data
        net_info = self.net_config
        if tensor_list == []:
            for node in net_info['model']['operator']:
                if node != 'input_data' and 'output' in net_info['model']['operator'][node].keys():
                    for tensor in net_info['model']['operator'][node]['output']:
                        net_info['model']['operator']['output_data']['input'][tensor] = {}
        else:
            for tensor in tensor_list:
                for node in net_info['model']['operator']:
                    operator = net_info['model']['operator']
                    if 'output' not in operator[node].keys():
                        continue
                    for tensor_name in operator[node]['output']:
                        search = re.search(tensor, tensor_name, re.I)
                        if search is not None:
                            net_info['model']['operator']['output_data']['input'][tensor_name] = {}

        return net_info

    # pybind engine executor
    def engine_init(self, net_info={}, weight_data=b""):
        import engine_py as dp
        if not weight_data:
            weight_data = self.weight_data
        if not net_info:
            net_info = self.net_config
        op_configs = []
        tensor_output = []
        tensor_input = []
        attr_map_list = []
        for node in net_info['model']['operator']:
            tensor_input.append([])
            tensor_output.append([])
            opeartor = net_info['model']['operator'][node]
            if 'input' in opeartor.keys():
                for input_name in opeartor['input']:
                    input_tensor = dp.tensor_config(input_name, [], "fp32", [], [])
                    tensor_input[-1].append(input_tensor)

            if 'output' in opeartor.keys():
                for (output_name, attrs) in opeartor['output'].items():
                    tensor_location = []
                    if 'location' in attrs.keys():
                        tensor_location = attrs['location']
                    tensor_strides = []
                    if "strides" in attrs.keys():
                        tensor_strides = attrs["strides"]
                    tensor_shape = []
                    if "shape" in attrs.keys():
                        tensor_shape = attrs["shape"]  
                    tensor_dtype = 'fp32'
                    if "dtype" in attrs.keys():
                        tensor_dtype = attrs["dtype"]
                    output_tensor = dp.tensor_config(output_name, tensor_shape, tensor_dtype,
                                                     tensor_strides, tensor_location)
                    tensor_output[-1].append(output_tensor)

            if 'attr' in opeartor.keys():
                op_attr = opeartor['attr']
                attr_maps = {}
                for (k, v) in op_attr.items():
                    attr_maps[str(k)] = str(v)
                attr_map_item = dp.attrs_config(attr_maps)
                attr_map_list.append(attr_map_item)
            else:
                attr_map = dp.attrs_config({})
                attr_map_list.append(attr_map)
            op_type = net_info['model']['operator'][node]['type']
            op_config = dp.op_config(str(node), str(op_type), tensor_input[-1], tensor_output[-1],
                                     attr_map_list[-1])
            op_configs.append(op_config)

        model_config = dp.model_config(net_info['model']['name'], op_configs)
        output_list = []
        for node in net_info['model']['operator']['output_data']['input']:
            output_list.append(node)
        model = dp.Model(model_config, weight_data)
        self._engine = [model, output_list, op_configs, tensor_output, tensor_input, attr_map_list]

    def inference(self, input_data):
        if self._engine is None:
            self.engine_init()
        output = self._engine[0].forward(input_data)
        index = 0
        output_dict = OrderedDict()
        for node in self._engine[1]:
            output_dict[node] = output[index]
            index += 1

        return output_dict

    def graph_init(self, config, weight_data=None):
        '''
        example:
                from engine.converter.graph import Graph
                newgraph = Graph()
                newgraph.graph_init('./ir/conf.yaml', './ir/model.bin')
                out = newgraph.inference([input_0, input_1, input_2])
        '''
        import yaml
        yamlPath = os.path.join(config)
        f = open(yamlPath, 'r', encoding='utf-8')
        cfg = f.read()
        d = yaml.load(cfg)
        if weight_data is None:
            weight_data = b""
        else:
            bin_file = open(weight_data, 'rb')
            weight_data = bin_file.read()

        self.engine_init(d, weight_data)


    def save(self, output_dir=None):
        logger.info("Start to emit the intermediate representation of model...")
        if output_dir is None:
            dir_name = os.getcwd()
            output_dir = os.path.join(dir_name, 'ir/')

        output_dir = os.path.abspath(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # set the bin and yaml file name
        bin_file = os.path.join(output_dir, 'model.bin')
        yaml_file = os.path.join(output_dir, 'conf.yaml')

        # serialize_weight
        weight_data = self.weight_data
        with open(bin_file, 'wb') as f:
            f.write(weight_data)

        # serialize_network
        net_info = self.net_config
        with open(yaml_file, "w", encoding="utf-8") as f:
            # for write list, no use '-' to split the list, which is the default action in yaml
            def list_representer(dumper, data):
                return dumper.represent_sequence(u'tag:yaml.org,2002:seq', data, flow_style=True)
            # for write OrderedDict

            def dict_representer(dumper, data):
                return dumper.represent_mapping("tag:yaml.org,2002:map", data.items())
            yaml.add_representer(list, list_representer)
            yaml.add_representer(OrderedDict, dict_representer)
            yaml.dump(net_info, f, default_flow_style=False, sort_keys=False)

        logger.info("Emit done...")
