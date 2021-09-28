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


@pattern_registry(pattern_type='LayerNormWithReduceMean')
class LayerNormWithReduceMean(Pattern):
    def __call__(self, model):

        pattern_mapping_config = {
            'LayerNormWithReduceMean': [
                {
                    'patterns': {
                        'in': [[(0, 'LayerNorm'), (1, 'ReduceMean')]],
                        'out': [[(0, 'LayerNorm'), (1, 'Reshape'), (2, 'ReduceMean'),
                                (3, 'Reshape')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 0,
                        1: 'reshape_before_reducemean',
                        2: 'reducemean_after_reshape',
                        3: 1
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }, {
                            0: [1]
                        }, {
                            0: [2]
                        }], [[0, 1, 2], 3]],
                        1: [[
                            {'input_data': [0]}
                        ], [[1], 2]],
                        2: [[], [[], 1]],
                        3: [[], [[], 1]]
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

        def _set_attr(hidden_size, epsilon, reduce_mean_attr, node_names, model):
            attr1 = OrderedDict()
            attr1['epsilon'] = float(epsilon)
            attr2 = OrderedDict()
            attr2['dst_shape'] = '-1,-1,' + str(hidden_size)
            attr2['dims'] = '0,1'
            attr3 = reduce_mean_attr
            attr4 = OrderedDict()
            attr4['dst_shape'] = '-1,' + str(hidden_size) 

            ln_node_idx = model.get_node_id(node_names[0])
            model.nodes[ln_node_idx].attr = attr1

            reshape_1_node_idx = model.get_node_id(node_names[1])
            model.nodes[reshape_1_node_idx].attr = attr2

            reduce_mean_node_idx = model.get_node_id(node_names[2])
            model.nodes[reduce_mean_node_idx].attr = attr3

            reshape_2_node_idx = model.get_node_id(node_names[3])
            model.nodes[reshape_2_node_idx].attr = attr4


        pattern_dict = pattern_mapping_config['LayerNormWithReduceMean'][0]
        model, new_node_names, ret_old_nodes = util.pattern_mapping(pattern_dict, model)
        if len(new_node_names) != 0:
            for j in range(len(new_node_names)):
                ln_node = ret_old_nodes[j][0]
                reduce_mean_node = ret_old_nodes[j][1]
                hidden_size = int(ln_node.input_tensors[-1].shape[-1])
                epsilon = ln_node.attr['epsilon']
                _set_attr(hidden_size, epsilon, reduce_mean_node.attr, new_node_names[j], model)
            
            return model
    
        return model
