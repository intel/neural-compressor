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
import copy
from .. import graph_utils as util


@pattern_registry(pattern_type='AttentionReshape')
class AttentionReshape(Pattern):
    def __call__(self, model):

        pattern_mapping_config = {
            'AttentionReshape': [
                {
                    'patterns': {
                        'in': [[(0, 'Mul'), (1, 'Pack'), (2, 'Reshape')]],
                        'out': [[(0, 'Reshape')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 2
                    },
                    'input_tensors': {
                        0: [[{
                            2: [0]
                        }], [[0], 1]]
                    },
                    'output_tensors': {
                        0: [[{
                            2: [0]
                        }], [[0], 1]]
                    },
                    'returns': [1]
                },

                # distil_bert_base
                {
                    'patterns': {
                        'in': [[(0, 'Unsqueeze'), (1, 'Concat'), (2, 'Reshape'), 
                                (3, 'MatMulWithBias')]],
                        'out': [[(0, 'Reshape'), (1, 'MatMulWithBias')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 2,
                        1: 3
                    },
                    'input_tensors': {
                        0: [[{
                            2: [0]
                        }], [[0], 1]],
                        1: [[
                            {3: [1]}, {3: [2]}
                        ], [[1, 2], 3]],
                    },
                    'output_tensors': {
                        0: [[{
                            2: [0]
                        }], [[0], 1]],
                        1: [[{
                            3: [0]
                        }], [[0], 1]]
                    },
                    'returns': [1, 3]
                },

                # geminet
                {
                    'patterns': {
                        'in': [[(0, 'Reshape'), (1, 'MatMulWithBias')]],
                        'out': [[(0, 'Reshape'), (1, 'MatMulWithBias')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 0,
                        1: 1
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }], [[0], 1]],
                        1: [[
                            {1: [1]}, {1: [2]}
                        ], [[1, 2], 3]],
                    },
                    'output_tensors': {
                        0: [[{
                            0: [0]
                        }], [[0], 1]],
                        1: [[{
                            1: [0]
                        }], [[0], 1]]
                    },
                    'returns': [0, 1]
                },
            ]
        }

        def _set_attr(hidden_size, node_names, model):
            attr = OrderedDict()
            attr['dst_shape'] = '-1,' + str(hidden_size)
            reshape_node_idx = model.get_node_id(node_names[0])
            model.nodes[reshape_node_idx].attr = attr
        
        for i in range(len(pattern_mapping_config['AttentionReshape'])-1):
            pattern_dict = pattern_mapping_config['AttentionReshape'][i]
            model, new_node_names, ret_old_nodes = util.pattern_mapping(pattern_dict, model)
            if len(new_node_names) != 0:
                for j in range(len(new_node_names)):
                    pack_node = ret_old_nodes[j][0]
                    hidden_size = int(pack_node.input_tensors[-1].data)
                    _set_attr(hidden_size, new_node_names[j], model)
                    if len(ret_old_nodes[j]) == 2:
                        assert ret_old_nodes[j][1].op_type == 'MatMulWithBias'
                        mat_node_idx = model.get_node_id(new_node_names[j][1])
                        model.nodes[mat_node_idx].attr = ret_old_nodes[j][1].attr
                
                return model
        
        # special reshape node, like has '0,0,768' dst_shape attr
        pattern_dict = pattern_mapping_config['AttentionReshape'][-1]
        model, new_node_names, ret_old_nodes = util.pattern_mapping(pattern_dict, model)
        if len(new_node_names) != 0:
            for j in range(len(new_node_names)):
                reshape_node = ret_old_nodes[j][0]
                try:
                    dst_shape = reshape_node.attr['dst_shape']
                except:
                    dst_shape = None
                if dst_shape != None and dst_shape.split(',')[0] == '0':
                    hidden_size = int(dst_shape.split(',')[-1])
                    _set_attr(hidden_size, new_node_names[j], model)
                mat_node_idx = model.get_node_id(new_node_names[j][1])
                model.nodes[mat_node_idx].attr = ret_old_nodes[j][1].attr
            
            return model
        
        return model
