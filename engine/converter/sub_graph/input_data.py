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


@pattern_registry(pattern_type='InputData')
class InputData(Pattern):
    def __call__(self, model):

        pattern_mapping_config = {
            'InputData': [
                {
                    'patterns': {
                        'in': [[(0, 'input_ids'), (1, 'segment_ids'), (2, 'input_mask')]],
                        'out': [[(0, 'Input')]]
                    },
                    'search_mode': 'node_name',
                    'node_names': {
                        0: 'input_data'
                    },
                    'input_tensors': {
                        0: [[], [[], 0]]
                    },
                    'output_tensors': {
                        0: [[{
                            0: [0]
                        }, {
                            1: [0]
                        }, {
                            2: [0]
                        }], [[0, 1, 2], 3]]
                    },
                    'returns': []
                },

                # bert_base
                {
                    'patterns': {
                        'in': [[(0, 'IteratorGetNext')]],
                        'out': [[(0, 'Input')]]
                    },
                    'search_mode': 'node_name',
                    'node_names': {
                        0: 'input_data'
                    },
                    'input_tensors': {
                        0: [[], [[], 0]]
                    },
                    'output_tensors': {
                        0: [[{
                            0: [0]
                        }, {
                            0: [4]
                        }, {
                            0: [1]
                        }], [[0, 1, 2], 3]]
                    },
                    'returns': []
                },
                
                # distil_bert_base
                {
                    'patterns': {
                        'in': [[(0, 'input_ids'), (1, 'attention_mask') ]],
                        'out': [[(0, 'Input')]]
                    },
                    'search_mode': 'node_name',
                    'node_names': {
                        0: 'input_data'
                    },
                    'input_tensors': {
                        0: [[], [[], 0]]
                    },
                    'output_tensors': {
                        0: [[{
                            0: [0]
                        }, {
                            1: [0]
                        }
                        ], [[0, 1], 2]]
                    },
                    'returns': []
                },
                
            ]
        }
        
        for i in range(len(pattern_mapping_config['InputData'])):
            pattern_dict = pattern_mapping_config['InputData'][i]
            model, new_node_names, ret_old_nodes = util.pattern_mapping(pattern_dict, model)
            if len(new_node_names) != 0:
                model.nodes[0].attr = None
                for j in range(len(model.nodes[0].output_tensors)):
                    model.nodes[0].output_tensors[j].shape = [-1, -1]
                
                return model
        
        return model