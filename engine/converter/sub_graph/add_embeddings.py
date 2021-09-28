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


@pattern_registry(pattern_type='AddEmbeddings')
class AddEmbeddings(Pattern):
    def __call__(self, model):

        pattern_mapping_config = {
            'AddEmbeddings': [
                {
                    'patterns': {
                        'in': [[(0, ['AddV2', 'Add']), (1, ['AddV2', 'Add']), (2, 'LayerNorm'),
                                (3, 'Reshape')]],
                        'out': [[(0, 'BinaryAdd'), (1, 'Reshape'), (2, 'Reshape'), 
                                (3, 'LayerNorm')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 0,
                        1: 'embeddings/after_add_reshape',
                        2: 'embeddings_add/reshape_2d',
                        3: 3
                    },
                    'input_tensors': {
                        0: [[{
                            0: [1]
                        }, {
                            1: [1]
                        }, {
                            0: [0]
                        }], [[0, 1, 2], 3]],
                        1: [[{
                            'input_data': [0]
                        }], [[1], 2]],
                        2: [[], [[], 1]],
                        3: [[{2: [1]}, {2: [2]}],
                            [[1, 2], 3]],
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[], [[], 1]],
                        2: [[], [[], 1]],
                        3: [[{
                            3: [0]
                        }], [[0], 1]]
                    },
                    'returns': [0, 2]
                },
                
                # geminet
                {
                    'patterns': {
                        'in': [[(0, ['AddV2', 'Add']), (1, ['AddV2', 'Add']), (2, 'LayerNorm')]],
                        'out': [[(0, 'BinaryAdd'), (1, 'Reshape'), (2, 'Reshape'), 
                                (3, 'LayerNorm')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 0,
                        1: 'embeddings/after_add_reshape',
                        2: 'embeddings_add/reshape_2d',
                        3: 2
                    },
                    'input_tensors': {
                        0: [[{
                            0: [1]
                        }, {
                            1: [1]
                        }, {
                            0: [0]
                        }], [[0, 1, 2], 3]],
                        1: [[{
                            'input_data': [0]
                        }], [[1], 2]],
                        2: [[], [[], 1]],
                        3: [[{2: [1]}, {2: [2]}], 
                            [[1, 2], 3]]
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[], [[], 1]],
                        2: [[], [[], 1]],
                        3: [[{
                            2: [0]
                        }], [[0], 1]]
                    },
                    'returns': [0, 2]
                },


                # distil_bert_base
                {
                    'patterns': {
                        'in': [[(0, ['AddV2', 'Add']), (1, 'LayerNorm')]],
                        'out': [[(0, 'BinaryAdd'), (1, 'Reshape'), (2, 'Reshape'), 
                                (3, 'LayerNorm')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 0,
                        1: 'embeddings/after_add_reshape',
                        2: 'embeddings_add/reshape_2d',
                        3: 1
                    },
                    'input_tensors': {
                        0: [[{
                            0: [1]
                        }, 
                        {
                            0: [0]
                        }], [[0, 1], 2]],
                        1: [[{
                            'input_data': [0]
                        }], [[1], 2]],
                        2: [[], [[], 1]],
                        3: [[{1: [1]}, {1: [2]}], 
                            [[1, 2], 3]]
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[], [[], 1]],
                        2: [[], [[], 1]],
                        3: [[{
                            1: [0]
                        }], [[0], 1]]
                    },
                    'returns': [0, 1]
                },
            ]
        }

        def _set_attr(hidden_size, epsilon, node_names, model):
            attr1 = OrderedDict()
            attr1['append_op'] = 'sum'
            attr2 = OrderedDict()
            attr2['dst_shape'] = '-1,-1,' + str(hidden_size)
            attr2['dims'] = '0,1'
            attr3 = OrderedDict()
            attr3['dst_shape'] = '-1,' + str(hidden_size)
            attr4 = OrderedDict()
            attr4['epsilon'] = float(epsilon)

            binary_add_node_idx = model.get_node_id(node_names[0])
            model.nodes[binary_add_node_idx].attr = attr1

            reshape_1_node_idx = model.get_node_id(node_names[1])
            model.nodes[reshape_1_node_idx].attr = attr2

            reshape_2_node_idx = model.get_node_id(node_names[2])
            model.nodes[reshape_2_node_idx].attr = attr3

            ln_node_idx = model.get_node_id(node_names[3])
            model.nodes[ln_node_idx].attr = attr4
        
        for i in range(len(pattern_mapping_config['AddEmbeddings'])):
            pattern_dict = pattern_mapping_config['AddEmbeddings'][i]
            model, new_node_names, ret_old_nodes = util.pattern_mapping(pattern_dict, model)
            if len(new_node_names) != 0:
                for j in range(len(new_node_names)):
                    add_node = ret_old_nodes[j][0]
                    pre_node = model.get_node_by_name(add_node.input_tensors[0].source_op[0])
                    hidden_size = int(pre_node.attr['dst_shape'].split(',')[-1])
                    ln_node = ret_old_nodes[j][1]
                    epsilon = ln_node.attr['epsilon']
                    _set_attr(hidden_size, epsilon, new_node_names[j], model)
                    if len(pattern_dict['patterns']['in'][0]) == 2:
                        binary_add_node_idx = model.get_node_id(new_node_names[j][0])
                        model.nodes[binary_add_node_idx].attr = OrderedDict()
                       
                return model
        
        return model
