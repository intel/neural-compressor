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


@pattern_registry(pattern_type='InteractFeatures')
class InteractFeatures(Pattern):
    def __call__(self, model):

        pattern_mapping_config = {
            'InteractFeatures': [
                # DLRM onnx model
                {
                    'patterns': {
                        'in': [[(0, 'Relu'), (1, 'Shape'), (2, 'Gather'), (3, 'Unsqueeze'), 
                                (7, 'Concat'), (9, 'Reshape')],
                               [(0, 'Relu'), (4, 'Shape'), (5, 'Gather'), (6, 'Unsqueeze'),
                                (7, 'Concat')],
                               [(0, 'Relu'), (8, 'Concat'), (9, 'Reshape')]],
                        'out': [[(0, 'Relu'), (1, 'Concat'), (2, 'Reshape')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 0,
                        1: 8,
                        2: 9,
                    },
                    'input_tensors': {
                        0: [[{0: [0]}], [[0], 1]],
                        1: [[], [[], 1]],
                        2: [[{1: [0]}], [[1], 2]], 
                    },
                    'output_tensors': {
                        0 : [[{0: [0]}], [[0], 1]],
                        1 : [[{8: [0]}], [[0], 1]],
                        2 : [[{9: [0]}], [[0], 1]],
                    },
                    'returns': [8, 9, 1, 4] # need modify the relu node source op
                },

                {
                    'patterns': {
                        'in': [[(0, 'MatMul'), (1, 'Shape'), (2, 'Gather'), (8, 'Concat'), \
                                (11, 'Reshape'), (13, 'Transpose'), (14, 'Reshape')],
                               [(1, 'Shape'), (3, 'Gather'), (4, 'Mul'), (5, 'Add'), \
                                (9, 'Gather'), (11, 'Reshape')],
                               [(0, 'MatMul'), (6, 'Transpose'), (7, 'Flatten'), (9, 'Gather')],
                               [(5, 'Add'), (10, 'Shape'), (12, 'Concat'), (14, 'Reshape')]
                        ],
                        'out': [[(0, 'Matmul'), (1, 'Reshape'), (2, 'Gather'), (3, 'Transpose')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 0,
                        1: 7,
                        2: 9,
                        3: 14,
                    },
                    'input_tensors': {
                        0: [[{0: [0]}, {0: [1]}], [[0,1], 2]],
                        1: [[{0: [0]}], [[1], 2]],
                        2: [[], [[], 1]],
                        3: [[], [[], 1]],
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[], [[], 1]],
                        2: [[], [[], 1]],
                        3: [[{14: [0]}], [[0], 1]],
                    },
                    'returns': [4, 5, 6, 9, 13]
                },

            ]
        }

        def _set_attr(concat_num, node_names, model):
            attr = OrderedDict()
            attr['dst_shape'] = '-1, ' + str(concat_num) + ', -1'
            attr['dims'] = '0, 1'
            reshape_node_idx = model.get_node_id(node_names[2])
            model.nodes[reshape_node_idx].attr = attr


        concat_num = None
        for i in range(len(pattern_mapping_config['InteractFeatures'])):
            pattern_dict = pattern_mapping_config['InteractFeatures'][i]
            model, new_node_names, ret_old_nodes = util.pattern_mapping("InteractFeatures",
                                                                        pattern_dict, model)

            if len(new_node_names) != 0 and i == 0:
                for j in range(len(new_node_names)):
                    concat_num = len(ret_old_nodes[j][0].input_tensors)
                    concat_node_idx = model.get_node_id(new_node_names[j][1])
                    model.nodes[concat_node_idx].input_tensors = ret_old_nodes[j][0].input_tensors
                    model.nodes[concat_node_idx].attr = ret_old_nodes[j][0].attr
                    _set_attr(concat_num, new_node_names[j], model)
                    relu_node_idx = model.get_node_id(new_node_names[j][0])
                    model.nodes[relu_node_idx].output_tensors[0].dest_op.remove(
                        ret_old_nodes[j][2].name)
                    model.nodes[relu_node_idx].output_tensors[0].dest_op.remove(
                        ret_old_nodes[j][3].name)
                
            elif len(new_node_names) != 0 and i == 1:
                for j in range(len(new_node_names)):
                    matmul_attr = OrderedDict()
                    matmul_attr['dst_perm'] = ret_old_nodes[j][2].attr['dst_perm']
                    matmul_node_idx = model.get_node_id(new_node_names[j][0])
                    model.nodes[matmul_node_idx].attr = matmul_attr
                    reshape_attr = OrderedDict()
                    reshape_attr['dst_shape'] = '-1, -1, -1'
                    reshape_attr['dims'] = '1, 1, 0'
                    reshape_attr['mul'] = '0, 1'
                    reshape_node_idx = model.get_node_id(new_node_names[j][1])
                    model.nodes[reshape_node_idx].attr = reshape_attr
                    gather_attr = OrderedDict()
                    gather_attr = ret_old_nodes[j][3].attr
                    gather_node_idx = model.get_node_id(new_node_names[j][2])
                    model.nodes[gather_node_idx].attr = gather_attr
                    gather_indices = concat_num * ret_old_nodes[j][0].input_tensors[0].data + \
                                     ret_old_nodes[j][1].input_tensors[0].data
                    ret_old_nodes[j][0].input_tensors[0].data = gather_indices
                    model.change_node_input_tensors(new_node_names[j][2], 0, \
                                        ret_old_nodes[j][0].input_tensors[0], mode='insert')
                    transpose_node_idx = model.get_node_id(new_node_names[j][3])
                    transpose_attr = OrderedDict()
                    transpose_attr = ret_old_nodes[j][4].attr
                    model.nodes[transpose_node_idx].attr = transpose_attr

        return model
