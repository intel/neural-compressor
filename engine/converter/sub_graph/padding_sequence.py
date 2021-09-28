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
from ..ops.tensor import Tensor


@pattern_registry(pattern_type='PaddingSequence')
class PaddingSequence(Pattern):
    def __call__(self, model):

        pattern_mapping_config = {
            'PaddingSequence': [
                {
                    'patterns': {
                        'in': [[(1, 'Shape'), (2, 'StridedSlice'), (5, 'Pack'), (6, 'Reshape'),
                                (7, 'ExpandDims'), (8, 'Sub'), (9, 'Mul'), (10, ['Add', 'AddV2'])],
                               [(), (3, 'Shape'), (4, 'StridedSlice'), (5, 'Pack')],
                               [(), (0, 'Cast'), (6, 'Reshape')]],
                        'out': [[(0, 'AddV2')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 10
                    },
                    'input_tensors': {
                        0: [[{
                            10: [0]
                        }, {
                            'padding_sequence': [0]
                        }], [[0, 1], 2]]
                    },
                    'output_tensors': {
                        0: [[{
                            10: [0]
                        }], [[0], 1]]
                    },
                    'returns': []
                },

                # bert_base_mrpc
                {
                    'patterns': {
                        'in': [[(0, 'Shape'), (1, 'StridedSlice'), (2, 'Pack'), (3, 'Fill'),
                                (7, 'Mul'), (8, 'ExpandDims'), (9, 'Sub'), (10, 'Mul'),
                                (11, ['Add', 'AddV2'])],
                               [(1, 'StridedSlice'), (4, 'Pack'), (5, 'Reshape'), (6, 'Cast'),
                                (7, 'Mul')]],
                        'out': [[(0, 'AddV2')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 11
                    },
                    'input_tensors': {
                        0: [[{
                            11: [0]
                        }, {
                            'padding_sequence': [0]
                        }], [[0, 1], 2]]
                    },
                    'output_tensors': {
                        0: [[{
                            11: [0]
                        }], [[0], 1]]
                    },
                    'returns': []
                },

                # geminet
                {
                    'patterns': {
                        'in': [[(0, 'Unsqueeze'), (1, 'Unsqueeze'), (2, 'Cast'), (3, 'Sub'),
                                (4, 'Mul'), (5, 'Add')]],
                        'out': [[(0, 'AddV2')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 5
                    },
                    'input_tensors': {
                        0: [[{
                            5: [0]
                        }, {
                            'padding_sequence': [0]
                        }], [[0, 1], 2]]
                    },
                    'output_tensors': {
                        0: [[{
                            5: [0]
                        }], [[0], 1]]
                    },
                    'returns': []
                },
                
                # distil_bert_base
                {
                    'patterns': {
                        'in': [[(1, 'Unsqueeze'), (3, 'Concat'), (4, 'Reshape'), (6, 'Expand'),
                                (7, 'Cast'), (8, 'Where')],
                                [(), (2, 'Unsqueeze'), (3, 'Concat')],
                                [(), (5, 'Shape'), (6, 'Expand')],
                                [(), (0, 'Equal'), (4, 'Reshape')]],
                        'out': [[(0, 'AddV2')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 8
                    },
                    'input_tensors': {
                        0: [[{
                            5: [0]
                        }, {
                            'padding_sequence': [0]
                        }], [[0, 1], 2]]
                    },
                    'output_tensors': {
                        0: [[{
                            8: [0]
                        }], [[0], 1]]
                    },
                    'returns': [2]
                },
            ]
        }

        def _make_padding_sequence_node(tensor_idx, hidden_size, model):
            node_name = 'padding_sequence'
            heads_num = int(hidden_size / 64)
            input_data = model.nodes[0]
            if len(input_data.output_tensors) > tensor_idx:
                input_tensors = [copy.deepcopy(input_data.output_tensors[tensor_idx])]
            else:
                input_tensors = [Tensor()]
            output_tensors = [Tensor(name=node_name + ':0', source_op=[node_name], dest_op=[])]
            attr = OrderedDict()
            attr['dst_shape'] = '-1,' + str(heads_num) + ',0,-1'
            attr['dims'] = 1
            padding_sequence_node = util.construct_node(node_name,
                                                        'PaddingSequence',
                                                        input_tensors=input_tensors,
                                                        output_tensors=output_tensors,
                                                        attr=attr)

            model.insert_nodes(1, [padding_sequence_node])

            return model

        pattern_dict = pattern_mapping_config['PaddingSequence'][0]
        model = _make_padding_sequence_node(2, 1024, model)
        model, new_node_names, ret_old_nodes = util.pattern_mapping(pattern_dict, model)
        if len(new_node_names) != 0:
            return model
        else:
            model.remove_nodes(['padding_sequence'])

        pattern_dict = pattern_mapping_config['PaddingSequence'][1]
        model = _make_padding_sequence_node(2, 768, model)
        model, new_node_names, ret_old_nodes = util.pattern_mapping(pattern_dict, model)
        if len(new_node_names) != 0:
            return model
        else:
            model.remove_nodes(['padding_sequence'])
        
        pattern_dict = pattern_mapping_config['PaddingSequence'][2]
        model = _make_padding_sequence_node(2, 768, model)
        model, new_node_names, ret_old_nodes = util.pattern_mapping(pattern_dict, model)
        if len(new_node_names) != 0:
            return model
        else:
            model.remove_nodes(['padding_sequence'])
        
        pattern_dict = pattern_mapping_config['PaddingSequence'][3]
        model = _make_padding_sequence_node(1, 768, model)
        model, new_node_names, ret_old_nodes = util.pattern_mapping(pattern_dict, model)
        if len(new_node_names) != 0:
            # remove shape+gather in distil_bert_base
            for i in range(len(ret_old_nodes)):
                gather_node_name = ret_old_nodes[i][0].input_tensors[0].source_op[0]
                gather_node = model.get_node_by_name(gather_node_name)
                shape_node_name = gather_node.input_tensors[0].source_op[0]
                model.remove_nodes([gather_node_name, shape_node_name])
            return model
        else:
            model.remove_nodes(['padding_sequence'])
        
        return model
