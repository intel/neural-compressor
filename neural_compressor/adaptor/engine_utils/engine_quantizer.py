#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import copy

class EngineQuantizer:
    def __init__(self, model, dataloader, iterations, q_config, op_types_to_quantize):
        self.model = model
        self.dataloader = dataloader
        self.iterations = iterations
        self.config = q_config
        self._quantize_op = op_types_to_quantize
        self._no_quantize_op = ["Reshape", "Reorder"]
        self._quantize_output_op = []
        self._quantize_input_op = []
        self._part_quantize_op = []

    def quantize_model(self):
        # 0. get min/max of tensor
        self._get_tensors_min_max()
        # 1. quant node weight
        self._quant_node()
        # 2. insert quantize before quant node
        self._insert_quantize_op()
        # 3. quant parttern fusion
        self._quant_pattern_fusion()
        # 4. insert quant info to each tensor
        self._insert_quantize_info()

        return self.model

    # 0. get min/max of tensor
    def _get_tensors_min_max(self):
        pass

    # 1. quant node weight
    def _quant_node(self):
        pass

    # 2. insert quantize before quant node
    def _insert_quantize_op(self):
        pass

    # 3. quant parttern fusion
    def _quant_pattern_fusion(self):
        for node in self.model.nodes:
            if (node.op_type in self._quantize_op and \
                    self.config[node.name.split('_quant')[0]] != 'fp32') \
                    or (node.op_type in self._quantize_output_op):
                is_quantize_fusion = True
                is_quantize_fusion, pre_quantize_op, quantize_op = self._is_quantize_fusion(
                    node, is_quantize_fusion)
                for idx, input_tensor in enumerate(node.input_tensors):
                    if 'append_op' in node.attr and \
                        (node.attr['append_op'] == 'sum' \
                        or node.attr['append_op'] == 'binary_add' ) \
                        and idx == len(node.input_tensors) - 1 and \
                        input_tensor.dtype == 'fp32':
                        is_quantize_fusion &= False
                        break
                if pre_quantize_op and pre_quantize_op.op_type in self._quantize_op and \
                    self.config[pre_quantize_op.name] == 'fp32':
                    is_quantize_fusion &= False
                if is_quantize_fusion:
                    self.model.remove_nodes([quantize_op.name])
                    pre_quantize_op.output_tensors[0] = quantize_op.output_tensors[0]
                    node.attr['output_dtype'] = quantize_op.attr['output_dtype']
    
    # 4. insert quant info to each tensor
    def _insert_quantize_info(self):
        pass

    # copy const for fall back
    def _copy_const(self):
        visted_const_tensor_name = []
        for node_id, node in enumerate(self.model.nodes):
            for tensor_id, input_tensor in enumerate(node.input_tensors):
                if not input_tensor.source_op:
                    if input_tensor.name not in visted_const_tensor_name:
                        visted_const_tensor_name.append(input_tensor.name)
                    else:
                        input_tensor_new = copy.deepcopy(input_tensor)
                        input_tensor_new.name = node.name + '_' + input_tensor.name
                        input_tensor_new.dest_op = [node.name]
                        self.model.nodes[node_id].input_tensors[tensor_id] = input_tensor_new

    def _remove_duplicate_quantize_op(self):
        visit_tensors = []
        nodes = copy.deepcopy(self.model.nodes)
        for node in nodes:
            if node.op_type == "Quantize":
                input_name = node.input_tensors[0].name
                if input_name in visit_tensors:
                    self.model.remove_nodes([node.name])
                else:    
                    visit_tensors.append(input_name) 

    # if quantize_output_op can fuse quantize
    def _is_quantize_fusion(self, node, is_qunatize_fusion):
        for output in node.output_tensors:
            for next_node_name in output.dest_op:
                next_node_idx = self.model.get_node_id(next_node_name)
                next_node = self.model.nodes[next_node_idx]
                if next_node.op_type == "Quantize":
                    is_qunatize_fusion &= True
                    return is_qunatize_fusion, node, next_node
                elif next_node.op_type in self._no_quantize_op:
                    return self._is_quantize_fusion(next_node, is_qunatize_fusion)
                # TODO
                # elif next_node.op_type in ['InnerProduct']:
                #     continue
                else:
                    is_qunatize_fusion &= False
                    return is_qunatize_fusion & False, None, None

        is_qunatize_fusion &= False
        return is_qunatize_fusion, None, None
