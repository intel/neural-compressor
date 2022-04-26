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


@pattern_registry(pattern_type='StartEndLogits')
class StartEndLogits(Pattern):
    def __call__(self, model):

        patterns = {
            'StartEndLogits': [
                # bert_large_sparse_model in squadv1 dataset
                [[(0, 'Transpose'), (1, 'Unpack'), (2, 'Identity')]],
                # bert_large_squad from huggingface in squadv1 dataset
                [[(0, 'Split'), (1, 'Squeeze')]],
            ]
        }
        
        # tf has reshape operations before split logits, so just remove this split things.
        # But in onnx, MatMul have implicit broadcasting mechanism, like [M K N] * [N K].
        # So before spliting, it has not reshape operations. We need to insert reshape op 
        # when we remove the split pattern in onnx since engine emits tensor of size 
        # [bs*seq_len, hidden_size] like tf.
        for i in range(len(patterns['StartEndLogits'])):
            in_pattern = patterns['StartEndLogits'][i]
            match_result = util.search_pattern(in_pattern, model)
            if len(match_result) != 0:
                for each_ret in match_result:
                    first_node_idx = model.get_node_id(each_ret[0])
                    first_node = model.nodes[first_node_idx]
                    model.remove_nodes(each_ret[:-1])
                    if i == 1:
                        input_tensors = [copy.deepcopy(first_node.input_tensors[0]),
                                        copy.deepcopy(model.nodes[0].output_tensors[0])]
                        output_tensors = [Tensor(name='logits', source_op=[first_node.name])]
                        pre_first_node = model.get_node_by_name(input_tensors[0].source_op[0])
                        assert pre_first_node.op_type == 'MatMulWithBias'
                        last_dim = pre_first_node.input_tensors[1].shape[-1]
                        attr = OrderedDict({'dst_shape': '-1,-1,' + str(last_dim),
                                            'dims': '0,1'})
                        reshape_node = util.construct_node(first_node.name, 'Reshape', 
                            input_tensors=input_tensors, output_tensors=output_tensors, attr=attr)
                        model.insert_nodes(first_node_idx, [reshape_node])

        return model
