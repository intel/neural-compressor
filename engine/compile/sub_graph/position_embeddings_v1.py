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


@pattern_registry(pattern_type='PositionEmbeddingsV1')
class PositionEmbeddingsV1(Pattern):
    def __call__(self, model):

        pattern_mapping_config = {
            'PositionEmbeddingsV1': [
                # roberta_base
                {
                    'patterns': {
                        'in': [[(0, 'Equal'), (1, 'Not'), (2, 'Cast'), (3, 'CumSum'), 
                                (4, 'Add'), (5, 'Mul'), (6, 'Cast'), (7, 'Add'), (8, 'Gather')]],
                        'out': [[(0, 'PositionIds'), (1, 'Reshape'),
                                 (2, 'Gather'), (3, 'Reshape'), (4, 'Reshape')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 'position_embeddings/Add_40',
                        1: 'position_embeddings/reshape',
                        2: 8,
                        3: 'position_embeddings/after/reshape',
                        4: 'position_embeddings/add_reshape',
                    },
                    'input_tensors': {
                        0: [[{
                            'input_data': [0]
                        }], [[0], 1]],
                        1: [[], [[], 1]],
                        2: [[{
                            8: [0]
                        }], [[1], 2]],
                        3: [[{
                            'input_data': [0]
                        }], [[1], 2]],
                        4: [[{
                            'input_data': [0]
                        }], [[1], 2]],
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[], [[], 1]],
                        2: [[], [[], 1]],
                        3: [[], [[], 1]],
                        4: [[{
                            8: [0]
                        }], [[0], 1]]
                    },
                    'returns': [8]
                },
            ]
        }

        def _set_attr(hidden_size, axis, batch_dims, node_names, model):
                attr0 = OrderedDict()
                attr0['mode'] = "roberta"
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
                
                position_node_idx = model.get_node_id(node_names[0])
                model.nodes[position_node_idx].attr = attr0

                reshape_0_node_idx = model.get_node_id(node_names[1])
                model.nodes[reshape_0_node_idx].attr = attr1

                gather_node_idx = model.get_node_id(node_names[2])
                model.nodes[gather_node_idx].attr = attr2

                reshape_1_node_idx = model.get_node_id(node_names[3])
                model.nodes[reshape_1_node_idx].attr = attr3

                reshape_2_node_idx = model.get_node_id(node_names[4])
                model.nodes[reshape_2_node_idx].attr = attr4

        # bert_roberta_base
        pattern_dict = pattern_mapping_config['PositionEmbeddingsV1'][0]
        model, new_node_names, ret_old_nodes = \
            util.pattern_mapping('PositionEmbeddingsV1', pattern_dict, model)
        if len(new_node_names) != 0:
            for i in range(len(new_node_names)):
                gatherv2_node = ret_old_nodes[i][0]
                hidden_size = int(gatherv2_node.input_tensors[0].shape[-1])
                axis = gatherv2_node.attr['axis']
                batch_dims = gatherv2_node.attr['batch_dims']
                _set_attr(hidden_size, axis, batch_dims, new_node_names[i], model)
                
            return model
        
        return model
