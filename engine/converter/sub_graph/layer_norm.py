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


@pattern_registry(pattern_type='LayerNorm')
class LayerNorm(Pattern):
    """
    Different model has the different layer_norm pattern,
    which means they have differrnt nodes order even though they all follow the formula.
    For exmaple:

    LayerNorm(src, gamma, beta)
        = gamma * (src-mean) / sqrt(variance+c) + beta  # 1, like transformer_lt
        = src * gamma / sqrt(variance+c) + (beta-mean*gamma/sqrt(variance+c)) # 2, like bert_large
    """
    def __call__(self, model):

        pattern_mapping_config = {
            'LayerNorm': [

                # bert_base has two layer_norm patterns
                {
                    'patterns': {
                        'in': [[(0, 'Shape'), (1, 'StridedSlice'), (2, 'Mul'), (4, 'Mul'),
                                (9, 'Pack'), (10, 'Fill'), (13, 'FusedBatchNormV3'),
                                (14, 'Reshape'), (15, 'Mul'), (16, ['Add', 'AddV2'])],
                               [(0, 'Shape'), (3, 'StridedSlice'), (4, 'Mul')],
                               [(0, 'Shape'), (5, 'StridedSlice'), (6, 'Mul'), (7, 'Pack'),
                                (8, 'Reshape'), (13, 'FusedBatchNormV3')],
                               [(4, 'Mul'), (11, 'Pack'), (12, 'Fill'), (13, 'FusedBatchNormV3')]],
                        'out': [[(0, 'LayerNorm')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 16
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }, {
                            15: [1]
                        }, {
                            16: [1]
                        }], [[0, 1, 2], 3]]
                    },
                    'output_tensors': {
                        0: [[{
                            16: [0]
                        }], [[0], 1]]
                    },
                    'returns': [13]
                },
                {
                    'patterns': {
                        'in': [[(0, 'Shape'), (1, 'StridedSlice'), (2, 'Mul'), (5, 'Pack'),
                                (6, 'Reshape'), (11, 'FusedBatchNormV3'), (12, 'Reshape'),
                                (13, 'Mul'), (14, ['Add', 'AddV2'])],
                               [(0, 'Shape'), (3, 'StridedSlice'), (4, 'Mul'), (5, 'Pack')],
                               [(2, 'Mul'), (7, 'Pack'), (8, 'Fill'), ((11, 'FusedBatchNormV3'))],
                               [(2, 'Mul'), (9, 'Pack'), (10, 'Fill'),
                                ((11, 'FusedBatchNormV3'))]],
                        'out': [[(0, 'LayerNorm')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 14
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }, {
                            13: [1]
                        }, {
                            14: [1]
                        }], [[0, 1, 2], 3]]
                    },
                    'output_tensors': {
                        0: [[{
                            14: [0]
                        }], [[0], 1]]
                    },
                    'returns': [11]
                },

                {
                    'patterns': {
                        'in': [[(0, 'Mean'), (1, 'SquaredDifference'), (2, 'Mean'),
                                (3, ['Add', 'AddV2']), (4, 'Rsqrt'), (5, 'Mul'), (7, 'Mul'),
                                (8, 'Sub'), (9, ['Add', 'AddV2'])],
                               [(5, 'Mul'), (6, 'Mul'), (9, ['Add', 'AddV2'])]],
                        'out': [[(0, 'LayerNorm')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 9
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }, {
                            5: [1]
                        }, {
                            8: [0]
                        }], [[0, 1, 2], 3]]
                    },
                    'output_tensors': {
                        0: [[{
                            9: [0]
                        }], [[0], 1]]
                    },
                    'returns': [3]
                },
                
                # distil_bert_base
                {
                    'patterns': {
                        'in': [[(0, 'ReduceMean'), (1, 'Sub'), (2, 'Pow'), (3, 'ReduceMean'), 
                                (4, 'Add'), (5, 'Sqrt'), (6, 'Div'), (7,'Mul'), (8, 'Add')]],
                        'out': [[(0, 'LayerNorm')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 8
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }, {
                            7: [1]
                        }, {
                            8: [1]
                        }], [[0, 1, 2], 3]]
                    },
                    'output_tensors': {
                        0: [[{
                            8: [0]
                        }], [[0], 1]]
                    },
                    'returns': [4]
                },
            ]
        }

        def _set_attr(epsilon, node_names, model):
            attr = OrderedDict()
            attr['epsilon'] = float(epsilon)
            # attr['output_dtype'] = 'fp32'
            ln_node_idx = model.get_node_id(node_names[0])
            model.nodes[ln_node_idx].attr = attr

        # bert_base layer_norm patterns
        pattern_dict = pattern_mapping_config['LayerNorm'][0]
        model, new_node_names, ret_old_nodes = util.pattern_mapping(pattern_dict, model)
        if len(new_node_names) != 0:
            for i in range(len(new_node_names)):
                epsilon = ret_old_nodes[i][0].attr['epsilon']
                _set_attr(epsilon, new_node_names[i], model)
            pattern_dict = pattern_mapping_config['LayerNorm'][1]
            model, new_node_names, ret_old_nodes = util.pattern_mapping(pattern_dict, model)
            assert len(new_node_names) != 0
            for i in range(len(new_node_names)):
                epsilon = ret_old_nodes[i][0].attr['epsilon']
                _set_attr(epsilon, new_node_names[i], model)

            return model
        
        for i in range(2, len(pattern_mapping_config['LayerNorm'])):
            pattern_dict = pattern_mapping_config['LayerNorm'][i]
            model, new_node_names, ret_old_nodes = util.pattern_mapping(pattern_dict, model)
            if len(new_node_names) != 0:
                for j in range(len(new_node_names)):
                    epsilon = ret_old_nodes[j][0].input_tensors[1].data
                    _set_attr(epsilon, new_node_names[j], model)
               
                return model
        
        return model