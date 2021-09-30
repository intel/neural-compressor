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


@pattern_registry(pattern_type='LastLayerShape')
class LastLayerShape(Pattern):
    def __call__(self, model):

        pattern_mapping_config = {
            'LastLayerShape': [
                {
                    'patterns': {
                        'in': [[(0, 'Pack'), (1, 'Reshape'), (2, 'Shape'), (3, 'StridedSlice'),
                                (5, 'Mul'), (6, 'Pack'), (7, 'Reshape'), (8, 'MatMulWithBias'),
                                (10, 'Reshape')], [(2, 'Shape'), (4, 'StridedSlice'), (5, 'Mul')],
                               [(3, 'StridedSlice'), (9, 'Pack'), (10, 'Reshape')]],
                        'out': [[(0, 'MatMulWithBias'), (1, 'Reshape')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 8,
                        1: 10
                    },
                    'input_tensors': {
                        0: [[{
                            1: [0]
                        }, {
                            8: [1]
                        }, {
                            8: [2]
                        }], [[0, 1, 2], 3]],
                        1: [[{
                            'input_data': [0]
                        }], [[1], 2]]
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[{
                            10: [0]
                        }], [[0], 1]]
                    },
                    'returns': [8, 9]
                },
                {
                    'patterns': {
                        'in': [[(0, 'Pack'), (1, 'Reshape'), (2, 'StridedSlice'), (3, 'Squeeze')]],
                        'out': [[(0, 'Reshape'), (1, 'StridedSlice'), (2, 'Reshape')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 1,
                        1: 2,
                        2: 3
                    },
                    'input_tensors': {
                        0: [[{
                            1: [0]
                        }, {
                            'input_data': [0]
                        }], [[0, 1], 2]],
                        1: [[], [[], 1]],
                        2: [[], [[], 1]]
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[], [[], 1]],
                        2: [[{
                            3: [0]
                        }], [[0], 1]]
                    },
                    'returns': [0, 2]
                },

                # roberta_base / distil_bert_base
                {
                    'patterns': {
                        'in': [[(0, 'LayerNorm'), (1, 'Gather'), (2, 'MatMulWithBias')]],
                        'out': [[(0, 'LayerNorm'), (1, 'Reshape'), (2, 'StridedSlice'),
                                (3, 'Reshape'), (4, 'MatMulWithBias')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 0,
                        1: 'last_layer_reshape',
                        2: 'last_layer_strided_slice',
                        3: 1,
                        4: 2
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]}, {0: [1]}, {0: [2]}
                        ], [[0, 1, 2], 3]],
                        1: [[{
                            'input_data': [0]
                        }], [[1], 2]],
                        2: [[], [[], 1]],
                        3: [[], [[], 1]],
                        4: [[
                            {2: [1]}, {2: [2]}
                        ], [[1, 2], 3]]
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[], [[], 1]],
                        2: [[], [[], 1]],
                        3: [[{1: [0]}], [[0], 1]],
                        4: [[{
                            2: [0]
                        }], [[0], 1]]
                    },
                    'returns': [0, 2]
                }
            ]
        }

        # bert_mlperf
        pattern_dict = pattern_mapping_config['LastLayerShape'][0]
        model, new_node_names, ret_old_nodes = util.pattern_mapping(pattern_dict, model)
        if len(new_node_names) != 0:
            for i in range(len(new_node_names)):
                mat_node = ret_old_nodes[i][0]
                pack_node = ret_old_nodes[i][1]
                last_dim = int(pack_node.input_tensors[-1].data)

                attr1 = mat_node.attr
                mat_node_idx = model.get_node_id(new_node_names[i][0])
                model.nodes[mat_node_idx].attr = attr1

                attr2 = OrderedDict()
                attr2['dst_shape'] = '-1,-1,' + str(last_dim)
                attr2['dims'] = '0,1'
                reshape_node_idx = model.get_node_id(new_node_names[i][1])
                model.nodes[reshape_node_idx].attr = attr2

            return model

        # bert_base_mrpc
        pattern_dict = pattern_mapping_config['LastLayerShape'][1]
        model, new_node_names, ret_old_nodes = util.pattern_mapping(pattern_dict, model)
        if len(new_node_names) != 0:
            for i in range(len(new_node_names)):
                pack_node = ret_old_nodes[i][0]
                strided_slice_node = ret_old_nodes[i][1]
                hidden_size = int(pack_node.input_tensors[-1].data)

                attr1 = OrderedDict()
                attr1['dst_shape'] = '-1,-1,' + str(hidden_size)
                attr1['dims'] = '0,1'
                reshape_0_node_idx = model.get_node_id(new_node_names[i][0])
                model.nodes[reshape_0_node_idx].attr = attr1

                attr2 = strided_slice_node.attr
                strided_slice_node_idx = model.get_node_id(new_node_names[i][1])
                model.nodes[strided_slice_node_idx].attr = attr2

                attr3 = OrderedDict()
                attr3['dst_shape'] = '-1,' + str(hidden_size)
                reshape_1_node_idx = model.get_node_id(new_node_names[i][2])
                model.nodes[reshape_1_node_idx].attr = attr3

            return model
        
        # roberta_base / distil_bert
        pattern_dict = pattern_mapping_config['LastLayerShape'][2]
        model, new_node_names, ret_old_nodes = util.pattern_mapping(pattern_dict, model)
        if len(new_node_names) != 0:
            for i in range(len(new_node_names)):
                ln_node = ret_old_nodes[i][0]
                mat_node = ret_old_nodes[i][1]
                hidden_size = int(ln_node.input_tensors[-1].shape[-1])
                
                ln_node_idx = model.get_node_id(new_node_names[i][0])
                model.nodes[ln_node_idx].attr = ln_node.attr
                reshape_0_node_idx = model.get_node_id(new_node_names[i][1])
                model.nodes[reshape_0_node_idx].attr = OrderedDict({
                    'dst_shape': '-1,-1,' + str(hidden_size), 'dims': '0,1'})
                strided_slice_node_idx = model.get_node_id(new_node_names[i][2])
                model.nodes[strided_slice_node_idx].attr = OrderedDict({
                    'begin_mask': 5, 'ellipsis_mask': 0, 'end_mask': 5, 'new_axis_mask': 0,
                    'shrink_axis_mask': 0, 'begin': '0,0,0',  'end': '0,1,0', 'strides': '1,1,1'})
                reshape_1_node_idx = model.get_node_id(new_node_names[i][3])
                model.nodes[reshape_1_node_idx].attr = OrderedDict({
                        'dst_shape': '-1,' + str(hidden_size)})
                mat_node_idx = model.get_node_id(new_node_names[i][4])
                model.nodes[mat_node_idx].attr = mat_node.attr

            return model
        
        return model
