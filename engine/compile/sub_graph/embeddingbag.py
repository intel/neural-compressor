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

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util


@pattern_registry(pattern_type='EmbeddingBag')
class EmbeddingBag(Pattern):
    def __call__(self, model):

        pattern_mapping_config = {
            'EmbeddingBag': [
                # DLRM onnx model
                {
                    'patterns': {
                        'in': [[(0, 'Shape'), (1, 'Gather'), (3, 'Unsqueeze'), (4, 'Concat'),
                                (5, 'Slice'), (6, 'Shape'), (7, 'Gather'), (8, 'Loop')],
                               [(), (2, 'Gather'), (4, 'Concat')]],
                        'out': [[(0, 'Gather'), (1, 'Reshape'), (2, 'EmbeddingBag')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 2,
                        1: 6,
                        2: 8,
                    },
                    'input_tensors': {
                        0: [[{
                         2: [1]}
                         , {2: [0]}], [[0,1], 2]],
                        1: [[], [[], 1]],
                        2: [[{
                         0: [0]}
                         , {8: [0]}], [[0,2], 3]],
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[], [[], 1]],
                        2 : [[{8: [0]}], [[0], 1]]
                    },
                    'returns': [2]
                },

                {
                    'patterns': {
                        'in': [[(0, 'Squeeze'), (0, 'EmbeddingBag')]],
                        'out': [[(0, 'Reshape'), (0, 'EmbeddingBag')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 0,
                        1: 1
                    },
                    'input_tensors': {
                        0: [[{0: [0]}], [[0], 1]],
                        1: [[{1: [1]}, {1: [2]}], [[1,2], 3]],
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1 : [[{1: [0]}], [[0], 1]]
                    },
                    'returns': []
                },

            ]
        }

        def _set_attr(hidden_size, axis, batch_dims, node_names, model):
            attr1 = OrderedDict()
            attr1['dst_shape'] = -1
            attr2 = OrderedDict()
            attr2['axis'] = axis
            attr2['batch_dims'] = batch_dims

            reshape_0_node_idx = model.get_node_id(node_names[1])
            model.nodes[reshape_0_node_idx].attr = attr1

            gather_node_idx = model.get_node_id(node_names[0])
            model.nodes[gather_node_idx].attr = attr2

        for i in range(len(pattern_mapping_config['EmbeddingBag'])):
            pattern_dict = pattern_mapping_config['EmbeddingBag'][i]
            model, new_node_names, ret_old_nodes = util.pattern_mapping("EmbeddingBag", 
                                                                        pattern_dict, model)

            if len(new_node_names) != 0 and i == 0:
                for j in range(len(new_node_names)):
                    gatherv2_node = ret_old_nodes[j][0]
                    hidden_size = int(gatherv2_node.input_tensors[1].shape[-1])
                    axis = gatherv2_node.attr['axis']
                    batch_dims = gatherv2_node.attr['batch_dims']
                    _set_attr(hidden_size, axis, batch_dims, new_node_names[j], model)
            elif len(new_node_names) != 0 and i == 1:
                for j in range(len(new_node_names)):
                    attr = OrderedDict()
                    attr['dst_shape'] = -1
                    reshape_node_idx = model.get_node_id(new_node_names[j][0])
                    model.nodes[reshape_node_idx].attr = attr


        return model
