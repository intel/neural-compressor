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


@pattern_registry(pattern_type='TokenTypeEmbeddings')
class TokenTypeEmbeddings(Pattern):
    def __call__(self, model):

        pattern_mapping_config = {
            'TokenTypeEmbeddings': [
                {
                    'patterns': {
                        'in': [[(0, 'Reshape'), (1, 'Gather'), (4, 'Reshape')],
                               [(), (2, 'StridedSlice'), (3, 'Pack'), (4, 'Reshape')]],
                        'out': [[(0, 'Reshape'), (1, 'Gather'), (2, 'Reshape'), (3, 'Reshape')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 0,
                        1: 1,
                        2: 'token_type_embeddings/after/reshape',
                        3: 4
                    },
                    'input_tensors': {
                        0: [[{
                            'input_data': [1]
                        }], [[0], 1]],
                        1: [[{
                            1: [0]
                        }], [[1], 2]],
                        2: [[{
                            'input_data': [1]
                        }], [[1], 2]],
                        3: [[{
                            'input_data': [1]
                        }], [[1], 2]]
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[], [[], 1]],
                        2: [[], [[], 1]],
                        3: [[{
                            4: [0]
                        }], [[0], 1]]
                    },
                    'returns': [1]
                },

                # bert_base_mrpc
                {
                    'patterns': {
                        'in': [[(0, 'Reshape'), (1, 'OneHot'), (2, 'MatMul'), (6, 'Reshape')],
                               [(), (3, 'Shape'), (4, 'StridedSlice'), (5, 'Pack'),
                                (6, 'Reshape')]],
                        'out': [[(0, 'Reshape'), (1, 'OneHot'), (2, 'MatMul'), (3, 'Reshape'),
                                 (4, 'Reshape')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 0,
                        1: 1,
                        2: 2,
                        3: 'token_type_embeddings/after/reshape',
                        4: 6
                    },
                    'input_tensors': {
                        0: [[{
                            'input_data': [1]
                        }], [[0], 1]],
                        1: [[], [[], 1]],
                        2: [[{
                            2: [1]
                        }], [[1], 2]],
                        3: [[{
                            'input_data': [1]
                        }], [[1], 2]],
                        4: [[{
                            'input_data': [1]
                        }], [[1], 2]]
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[], [[], 1]],
                        2: [[], [[], 1]],
                        3: [[], [[], 1]],
                        4: [[{
                            6: [0]
                        }], [[0], 1]]
                    },
                    'returns': [1, 2]
                },

                # geminet
                {
                    'patterns': {
                        'in': [[(0, 'Input'), (1, 'Gather'), (2, 'Add'), (3, 'LayerNorm')]],
                        'out': [[(0, 'Reshape'), (1, 'Gather'), (2, 'Reshape'), (3, 'Reshape'),
                                (4, 'Add'), (5, 'LayerNorm')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 'token_type_embeddings/reshape',
                        1: 1,
                        2: 'token_type_embeddings/after/reshape',
                        3: 'token_type_embeddings/add_reshape',
                        4: 2,
                        5: 3,
                    },
                    'input_tensors': {
                        0: [[{
                            'input_data': [1]
                        }], [[0], 1]],
                        1: [[{
                            1: [0]
                        }], [[1], 2]],
                        2: [[{
                            'input_data': [1]
                        }], [[1], 2]],
                        3: [[{
                            'input_data': [1]
                        }], [[1], 2]],
                        4: [[{
                            2:[0, 1]
                        }], [[0], 2]],
                        5: [[
                            {3: [1]}, {3: [2]}
                        ], [[1, 2], 3]],
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
                        }], [[0], 1]],
                    },
                    'returns': [0, 1, 3]
                },
            ]
        }

        def _set_attr_with_gather(hidden_size, axis, batch_dims, node_names, model):
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

        pattern_dict = pattern_mapping_config['TokenTypeEmbeddings'][0]
        model, new_node_names, ret_old_nodes = util.pattern_mapping(pattern_dict, model)
        if len(new_node_names) != 0:
            for i in range(len(new_node_names)):
                gatherv2_node = ret_old_nodes[i][0]
                hidden_size = int(gatherv2_node.input_tensors[0].shape[-1])
                axis = gatherv2_node.attr['axis']
                batch_dims = gatherv2_node.attr['batch_dims']
                _set_attr_with_gather(hidden_size, axis, batch_dims, new_node_names[i], model)

            return model

        def _set_attr_with_onehot(hidden_size, one_hot_attr, mat_attr, node_names, model):
            attr1 = OrderedDict()
            attr1['dst_shape'] = -1
            attr2 = one_hot_attr
            attr3 = mat_attr
            attr4 = OrderedDict()
            attr4['dst_shape'] = '-1,-1,' + str(hidden_size)
            attr4['dims'] = '0,1'
            attr5 = OrderedDict()
            attr5['dst_shape'] = '-1,-1,' + str(hidden_size)
            attr5['dims'] = '0,1'
            attr5['mul'] = '1,2'
            for _ in range(len(node_names)):
                reshape_0_node_idx = model.get_node_id(node_names[0])
                model.nodes[reshape_0_node_idx].attr = attr1

                onehot_node_idx = model.get_node_id(node_names[1])
                model.nodes[onehot_node_idx].attr = attr2

                mat_node_idx = model.get_node_id(node_names[2])
                model.nodes[mat_node_idx].attr = attr3

                reshape_1_node_idx = model.get_node_id(node_names[3])
                model.nodes[reshape_1_node_idx].attr = attr4

                reshape_2_node_idx = model.get_node_id(node_names[4])
                model.nodes[reshape_2_node_idx].attr = attr5

        # bert_base onehot+matmul embeddings
        pattern_dict = pattern_mapping_config['TokenTypeEmbeddings'][1]
        model, new_node_names, ret_old_nodes = util.pattern_mapping(pattern_dict, model)
        if len(new_node_names) != 0:
            for i in range(len(new_node_names)):
                onehot_node = ret_old_nodes[i][0]
                mat_node = ret_old_nodes[i][1]
                hidden_size = int(mat_node.input_tensors[1].shape[-1])
                transpose_a = mat_node.attr['transpose_a']
                transpose_b = mat_node.attr['transpose_b']
                mat_attr = OrderedDict()
                if transpose_a:
                    mat_attr['src0_perm'] = '1,0'
                # see OneDNN InnerProduct related requirements
                if not transpose_b:
                    mat_attr['src1_perm'] = '1,0'
                else:
                    mat_attr['src1_perm'] = '0,1'
                _set_attr_with_onehot(hidden_size, onehot_node.attr, mat_attr, new_node_names[i],
                                      model)

            return model
        
        # geminet
        pattern_dict = pattern_mapping_config['TokenTypeEmbeddings'][2]
        model, new_node_names, ret_old_nodes = util.pattern_mapping(pattern_dict, model)
        if len(new_node_names) != 0:
            for i in range(len(new_node_names)):
                gatherv2_node = ret_old_nodes[i][1]
                hidden_size = int(gatherv2_node.input_tensors[0].shape[-1])
                axis = gatherv2_node.attr['axis']
                batch_dims = gatherv2_node.attr['batch_dims']
                _set_attr_with_gather(hidden_size, axis, batch_dims, new_node_names[i], model)

                ln_node = ret_old_nodes[i][2]
                ln_1_node_idx = model.get_node_id(new_node_names[i][5])
                model.nodes[ln_1_node_idx].attr = ln_node.attr
                model.insert_nodes(0, [ret_old_nodes[i][0]])
                                
            return model
        
        return model