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
import copy


@pattern_registry(pattern_type='LastHiddenStatesConcat')
class LastHiddenStatesConcat(Pattern):
    def __call__(self, model):

        pattern_mapping_config = {
            'LastHiddenStatesConcat': [
                {
                    'patterns': {
                        'in': [[(0, 'Concat'), (1, 'ReduceMean'), (2, 'MatMulWithBias')]],
                        'out': [[(0, 'Concat'), (1, 'ReduceMean'), (2, 'Reshape'), 
                                (3, 'MatMulWithBias'), (4, 'Reshape')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 0,
                        1: 1,
                        2: 'reshape_after_reducemean',
                        3: 2,
                        4: 'reshape_for_output',
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }, {
                            0: [1]
                        }], [[0, 1], 2]],
                        1: [[], [[], 1]],
                        2: [[], [[], 1]],
                        3: [[
                            {2: [1]},
                            {2: [2]}
                        ], [[1, 2], 3]],
                        4: [[
                            {'input_data': [0]}
                        ], [[1], 2]]
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[], [[], 1]],
                        2: [[], [[], 1]],
                        3: [[], [[], 1]],
                        4: [[{
                            2: [0]
                        }], [[0], 1]]
                    },
                    'returns': [0, 1, 2]
                },
            ]
        }

        def _set_attr(concat_attr, reduce_mean_attr, hidden_size, mat_attr, class_num, 
                        node_names, model):
            concat_node_idx = model.get_node_id(node_names[0])
            model.nodes[concat_node_idx].attr = concat_attr
            reduce_mean_node_idx = model.get_node_id(node_names[1])
            model.nodes[reduce_mean_node_idx].attr = reduce_mean_attr
            reshape_1_node_idx = model.get_node_id(node_names[2])
            model.nodes[reshape_1_node_idx].attr = OrderedDict({
                                                    'dst_shape': '-1,' + str(hidden_size)})
            mat_node_idx = model.get_node_id(node_names[3])
            model.nodes[mat_node_idx].attr = mat_attr
            reshape_2_node_idx = model.get_node_id(node_names[4])
            model.nodes[reshape_2_node_idx].attr = OrderedDict({
                                                    'dst_shape': '-1,-1,' + str(class_num),
                                                    'dims': '0,1'})


        pattern_dict = pattern_mapping_config['LastHiddenStatesConcat'][0]
        model, new_node_names, ret_old_nodes = util.pattern_mapping("LastHiddenStatesConcat", 
                                                                    pattern_dict, model)
        for j in range(len(new_node_names)):
            concat_attr = copy.deepcopy(ret_old_nodes[j][0].attr)
            reduce_mean_attr = copy.deepcopy(ret_old_nodes[j][1].attr)
            mat_node = ret_old_nodes[j][2]
            mat_attr = copy.deepcopy(mat_node.attr)
            hidden_size = int(mat_node.input_tensors[1].shape[0])
            class_num = int(mat_node.input_tensors[1].shape[1])
            _set_attr(concat_attr, reduce_mean_attr, hidden_size, mat_attr, class_num, 
                        new_node_names[j], model)
            
            # change the Unsqueeze nodes before Concat node to Reshape
            unsqueeze_nodes_names = model.get_pre_node_names(new_node_names[j][0])
            for name in unsqueeze_nodes_names:
                node_idx = model.get_node_id(name)
                model.nodes[node_idx].op_type = 'Reshape'
                shape_ref = model.nodes[0].output_tensors[0]
                model.change_node_input_tensors(name, 1, tensor=shape_ref, mode='insert')
                model.nodes[node_idx].attr =OrderedDict({
                                    'dst_shape': '-1,-1,' + str(hidden_size) + ',1',
                                    'dims': '0,1'})

        return model
