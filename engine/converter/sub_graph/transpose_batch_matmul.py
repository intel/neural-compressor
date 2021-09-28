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
import copy


@pattern_registry(pattern_type='TransposeBatchMatMul')
class TransposeBatchMatMul(Pattern):
    def __call__(self, model):

        pattern_mapping_config = {
            'TransposeBatchMatMul': [
                {
                'patterns': {
                    'in': [[(0, 'Transpose'), (2, ['BatchMatMul', 'BatchMatMulV2']), (3, 'Mul'),
                            (4, ['AddV2', 'Add'])],
                           [(), (1, 'Transpose'), (2, ['BatchMatMul', 'BatchMatMulV2'])]],
                    'out': [[(0, 'TransposeBatchMatMul')]]
                },
                'search_mode': 'op_type',
                'node_names': {
                    0: 4
                },
                'input_tensors': {
                    0: [[{
                        0: [0]
                    }, {
                        1: [0]
                    }, {
                        4: [0, 1]
                    }], [[0, 1, 2], 3]]
                },
                'output_tensors': {
                    0: [[{
                        4: [0]
                    }], [[0], 1]]
                },
                'returns': [0, 1, 2, 3]
            }, 

            # distil_bert_base
            {
                'patterns': {
                    'in': [[(0, 'Transpose'), (1, 'Div'), (3, 'MatMul'), (4, ['AddV2', 'Add'])],
                           [(), (2, 'Transpose'), (3, 'MatMul')]],
                    'out': [[(0, 'TransposeBatchMatMul')]]
                },
                'search_mode': 'op_type',
                'node_names': {
                    0: 4
                },
                'input_tensors': {
                    0: [[{
                        0: [0]
                    }, {
                        2: [0]
                    }, {
                        4: [0, 1]
                    }], [[0, 1, 2], 3]]
                },
                'output_tensors': {
                    0: [[{
                        4: [0]
                    }], [[0], 1]]
                },
                'returns': [0, 2, 3, 1]
            }, 

            # geminet
            {
                'patterns': {
                    'in': [[(0, 'Transpose'), (2, 'FusedMatMul'), (3, ['AddV2', 'Add'])],
                           [(), (1, 'Transpose'), (2, 'FusedMatMul')]],
                    'out': [[(0, 'TransposeBatchMatMul')]]
                },
                'search_mode': 'op_type',
                'node_names': {
                    0: 3
                },
                'input_tensors': {
                    0: [[{
                        0: [0]
                    }, {
                        1: [0]
                    }, {
                        3: [0, 1]
                    }], [[0, 1, 2], 3]]
                },
                'output_tensors': {
                    0: [[{
                        3: [0]
                    }], [[0], 1]]
                },
                'returns': [0, 1, 2]
            }, 
            
            {
                'patterns': {
                    'in': [[(0, 'Transpose'), (1, ['BatchMatMul', 'BatchMatMulV2', 'MatMul']),
                            (2, 'Transpose')]],
                    'out': [[(0, 'TransposeBatchMatMul')]]
                },
                'search_mode': 'op_type',
                'node_names': {
                    0: 2
                },
                'input_tensors': {
                    0: [[{
                        1: [0]
                    }, {
                        0: [0]
                    }], [[0, 1], 2]]
                },
                'output_tensors': {
                    0: [[{
                        2: [0]
                    }], [[0], 1]]
                },
                'returns': [0, 1, 2]
            },
            
            ]
        }

        def _adj_perm(src_perm):
            perm = copy.deepcopy(list(src_perm))
            _x = perm[-2:]
            ret = perm[:-2] + _x[::-1]
            return ret
        
        for i in range(0, len(pattern_mapping_config['TransposeBatchMatMul'])-1):
            pattern_dict = pattern_mapping_config['TransposeBatchMatMul'][i]
            model, new_node_names, ret_old_nodes = util.pattern_mapping(pattern_dict, model)
            if len(new_node_names) != 0:
                for j in range(len(new_node_names)):
                    transpose_list = []
                    transpose_list.append(util.str2list(ret_old_nodes[j][0].attr['dst_perm']))
                    transpose_list.append(util.str2list(ret_old_nodes[j][1].attr['dst_perm']))
                    transpose_a = ret_old_nodes[j][2].attr['transpose_a']
                    transpose_b = ret_old_nodes[j][2].attr['transpose_b']
                    if transpose_a:
                        transpose_list[0] = _adj_perm(transpose_list[0])
                    if transpose_b:
                        transpose_list[1] = _adj_perm(transpose_list[1])

                    attr = OrderedDict()
                    attr['src0_perm'] = util.list2str(transpose_list[0])
                    attr['src1_perm'] = util.list2str(transpose_list[1])
                    if len(ret_old_nodes[j]) == 3 \
                        and ret_old_nodes[j][2].op_type in ['FusedMatMul']:
                        output_scale = ret_old_nodes[j][2].attr['alpha']
                    else:
                        if ret_old_nodes[j][3].op_type == 'Div':
                            output_scale = 1.0 / ret_old_nodes[j][3].input_tensors[1].data
                        else:
                            output_scale = ret_old_nodes[j][3].input_tensors[1].data
                    attr['output_scale'] = float(output_scale)
                    attr['format_any'] = False
                    attr['append_op'] = 'binary_add'
                    tb_node_idx = model.get_node_id(new_node_names[j][0])
                    model.nodes[tb_node_idx].attr = attr
        
        pattern_dict = pattern_mapping_config['TransposeBatchMatMul'][-1]
        model, new_node_names, ret_old_nodes = util.pattern_mapping(pattern_dict, model)
        if len(new_node_names) != 0:
            for i in range(len(new_node_names)):
                transpose_list = []
                attr = OrderedDict()
                transpose_list.append(util.str2list(ret_old_nodes[i][0].attr['dst_perm']))
                transpose_list.append(util.str2list(ret_old_nodes[i][2].attr['dst_perm']))
                transpose_a = ret_old_nodes[i][1].attr['transpose_a']
                transpose_b = ret_old_nodes[i][1].attr['transpose_b']
                m_x = ret_old_nodes[i][1].input_tensors[0]

                if transpose_a:
                    src0_perm = [i for i in range(len(m_x.shape))]
                    src0_perm = _adj_perm(src0_perm)
                    attr['src0_perm'] = util.list2str(src0_perm)
                if transpose_b:
                    transpose_list[0] = _adj_perm(transpose_list[0])
                attr['src1_perm'] = util.list2str(transpose_list[0])
                attr['dst_perm'] = util.list2str(transpose_list[1])

                tb_node_idx = model.get_node_id(new_node_names[i][0])
                model.nodes[tb_node_idx].attr = attr
        
        
        return model
