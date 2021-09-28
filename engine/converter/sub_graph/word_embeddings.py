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


@pattern_registry(pattern_type='WordEmbeddings')
class WordEmbeddings(Pattern):
    def __call__(self, model):

        pattern_mapping_config = {
            'WordEmbeddings': [
                {
                    'patterns': {
                        'in': [[(0, 'ExpandDims'), (3, 'Shape'), (4, 'StridedSlice'), (6, 'Pack'),
                                (7, 'Reshape')], [(3, 'Shape'), (5, 'StridedSlice'), (6, 'Pack')],
                               [(0, 'ExpandDims'), (1, 'Reshape'), (2, 'Gather'), (7, 'Reshape')]],
                        'out': [[(0, 'Reshape'), (1, 'Gather'), (2, 'Reshape'), (3, 'Reshape')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 1,
                        1: 2,
                        2: 'word_embeddings/after/reshape',
                        3: 7
                    },
                    'input_tensors': {
                        0: [[{
                            'input_data': [0]
                        }], [[0], 1]],
                        1: [[{
                            2: [0]
                        }], [[1], 2]],
                        2: [[{
                            'input_data': [0]
                        }], [[1], 2]],
                        3: [[{
                            'input_data': [0]
                        }], [[1], 2]]
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[], [[], 1]],
                        2: [[], [[], 1]],
                        3: [[{
                            7: [0]
                        }], [[0], 1]]
                    },
                    'returns': [2]
                },

                # bert_base
                {
                    'patterns': {
                        'in': [[(0, 'ExpandDims'), (1, 'Reshape'), (2, 'Gather'), (6, 'Reshape')],
                               [(0, 'ExpandDims'), (3, 'Shape'), (4, 'StridedSlice'), (5, 'Pack'),
                                (6, 'Reshape')]],
                        'out': [[(0, 'Reshape'), (1, 'Gather'), (2, 'Reshape'), (3, 'Reshape')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 1,
                        1: 2,
                        2: 'word_embeddings/after/reshape',
                        3: 6
                    },
                    'input_tensors': {
                        0: [[{
                            'input_data': [0]
                        }], [[0], 1]],
                        1: [[{
                            2: [0]
                        }], [[1], 2]],
                        2: [[{
                            'input_data': [0]
                        }], [[1], 2]],
                        3: [[{
                            'input_data': [0]
                        }], [[1], 2]]
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[], [[], 1]],
                        2: [[], [[], 1]],
                        3: [[{
                            6: [0]
                        }], [[0], 1]]
                    },
                    'returns': [2]
                },
                
                # geminet
                {
                    'patterns': {
                        'in': [[(0, 'Input'), (1, 'Gather'), (2, 'Add'), (3, 'Add')]],
                        'out': [[(0, 'Reshape'), (1, 'Gather'), (2, 'Reshape'), 
                                (3, 'Reshape'), (4, 'Add'), (5, 'Add')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 'word_embeddings/reshape',
                        1: 1,
                        2: 'word_embeddings/after/reshape',
                        3: 'word_embeddings/add_reshape',
                        4: 2,
                        5: 3
                    },
                    'input_tensors': {
                        0: [[{
                            'input_data': [0]
                        }], [[0], 1]],
                        1: [[{
                            1: [0]
                        }], [[1], 2]],
                        2: [[{
                            'input_data': [0]
                        }], [[1], 2]],
                        3: [[{
                            'input_data': [0]
                        }], [[1], 2]],
                        4: [[
                            {2: [0, 1]}],
                        [[1], 2]],
                        5: [[
                            {3: [0,1]}],
                        [[1], 2]],
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[], [[], 1]],
                        2: [[], [[], 1]],
                        3: [[], [[], 1]],
                        4: [[{
                            2: [0]
                        }], [[0], 1]],
                        5: [[{
                            3: [0]
                        }], [[0], 1]]
                    },
                    'returns': [1, 0]
                },
                  
                # distil_bert_base
                {
                    'patterns': {
                        'in': [[(0, 'Input'), (1, 'Gather')]],
                        'out': [[(0, 'Reshape'), (1, 'Gather'), (2, 'Reshape'), 
                                (3, 'Reshape')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 'word_embeddings/reshape',
                        1: 1,
                        2: 'word_embeddings/after/reshape',
                        3: 'word_embeddings/add_reshape'
                    },
                    'input_tensors': {
                        0: [[{
                            'input_data': [0]
                        }], [[0], 1]],
                        1: [[{
                            1: [0]
                        }], [[1], 2]],
                        2: [[{
                            'input_data': [0]
                        }], [[1], 2]],
                        3: [[{
                            'input_data': [0]
                        }], [[1], 2]]
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[], [[], 1]],
                        2: [[], [[], 1]],
                        3: [[{
                            1: [0]
                        }], [[0], 1]],
                    },
                    'returns': [1, 0]
                },
               
            ]
        }

        def _set_attr(hidden_size, axis, batch_dims, node_names, model):
            attr1 = OrderedDict()
            attr1['dst_shape'] = -1
            attr2 = OrderedDict()
            attr2['axis'] = axis
            attr2['batch_dims'] = batch_dims
            attr3 = OrderedDict()
            attr3['dst_shape'] = '-1,-1,' + str(hidden_size)
            attr3['dims'] = '0,1'
            attr4 = OrderedDict()
            attr4['dst_shape'] = '-1,-1,' + str(hidden_size)
            attr4['dims'] = '0,1'
            attr4['mul'] = '1,2'

            reshape_0_node_idx = model.get_node_id(node_names[0])
            model.nodes[reshape_0_node_idx].attr = attr1

            gather_node_idx = model.get_node_id(node_names[1])
            model.nodes[gather_node_idx].attr = attr2

            reshape_1_node_idx = model.get_node_id(node_names[2])
            model.nodes[reshape_1_node_idx].attr = attr3

            reshape_2_node_idx = model.get_node_id(node_names[3])
            model.nodes[reshape_2_node_idx].attr = attr4

        for i in range(len(pattern_mapping_config['WordEmbeddings'])):
            pattern_dict = pattern_mapping_config['WordEmbeddings'][i]
            model, new_node_names, ret_old_nodes = util.pattern_mapping(pattern_dict, model)
            if len(new_node_names) != 0:
                for j in range(len(new_node_names)):
                    gatherv2_node = ret_old_nodes[j][0]
                    hidden_size = int(gatherv2_node.input_tensors[0].shape[-1])
                    axis = gatherv2_node.attr['axis']
                    batch_dims = gatherv2_node.attr['batch_dims']
                    _set_attr(hidden_size, axis, batch_dims, new_node_names[j], model)
                    if len(ret_old_nodes[j]) == 2:
                        assert ret_old_nodes[j][1].op_type == 'Input'
                        model.insert_nodes(0, [ret_old_nodes[j][1]])
                
                return model
        
        return model