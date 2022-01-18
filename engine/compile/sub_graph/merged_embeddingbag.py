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


@pattern_registry(pattern_type='MergedEmbeddingbag')
class MergedEmbeddingbag(Pattern):
    def __call__(self, model):

        pattern_mapping_config = {
            'MergedEmbeddingbag': [
                # DLRM onnx model
                {
                    'patterns': {
                        'in': [[(0, 'Split'), (1, 'Squeeze'), (2, 'Shape'),
                                (3, 'Gather'), (5, 'Unsqueeze'), (6, 'Concat'),
                                (7, 'Slice'), (8, 'Shape'), (9, 'Gather'), (10, 'Loop')],
                               [(), (4, 'Gather'), (6, 'Concat')]],
                    },
                },
            ]
        }

        pattern = pattern_mapping_config['MergedEmbeddingbag'][0]['patterns']['in']
        patterns_nodes_name = util.search_pattern(pattern, model)

        if len(patterns_nodes_name) != 0:
            split_idx = model.get_node_id(patterns_nodes_name[0][0])
            split_node = model.nodes[split_idx]
            gather_idx = model.get_node_id(patterns_nodes_name[0][4])
            gather_node = model.nodes[gather_idx]
            match_nodes_name = []
            merged_embeddingbag_inputs = []
            merged_embeddingbag_outputs = []

            merged_embeddingbag_inputs.append(gather_node.input_tensors[0])
            merged_embeddingbag_inputs.append(split_node.input_tensors[0])

            for pattern_nodes_name in patterns_nodes_name:
                loop_idx = model.get_node_id(pattern_nodes_name[10])
                loop_node = model.nodes[loop_idx]
                weight = loop_node.input_tensors[0]
                merged_embeddingbag_inputs.append(weight)
                loop_output = loop_node.output_tensors[0]
                merged_embeddingbag_outputs.append(loop_output)
                for idx in range(len(pattern_nodes_name)-1):
                    match_nodes_name.append(pattern_nodes_name[idx])

            merged_embeddingbag_node_name = "MergedEmbeddingbag_" + split_node.name
            merged_embeddingbag_node = util.construct_node(merged_embeddingbag_node_name,
            'MergedEmbeddingbag', merged_embeddingbag_inputs, merged_embeddingbag_outputs)

            model.remove_nodes(match_nodes_name)
            model.insert_nodes(split_idx, [merged_embeddingbag_node])

        return model
