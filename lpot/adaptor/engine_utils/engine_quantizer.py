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
import os
import pickle
import struct
from pathlib import Path
from engine.converter.ops.op import OPERATORS, Tensor
import numpy as np
from collections import namedtuple
from collections import OrderedDict

import copy

class EngineQuantizer:
    def __init__(self, model, dataloader, iterations, q_config, op_types_to_quantize):
        self.model = model
        self.dataloader = dataloader
        self.iterations = iterations
        self.config = q_config
        self._tensors_min_max = {}
        self._quantize_op = op_types_to_quantize
        self._quantize_input_op = ["Quantize"]
        self._quantize_output_op = ["Softmax"]
        self._part_quantize_op = self._quantize_input_op + self._quantize_output_op
        self._no_quantize_op = ["Reshape", "Reorder"]
        
    def quantize_model(self):
        # 0. get min/max of tensor
        self._get_tensors_min_max()
        # 1. quant innerproduct weight and bias
        self._quantize_innerproduct()
        # 2. insert quantize before innerproduct
        self._insert_quantize_op()
        # 3. insert quant info to each tensor
        self._insert_quantize_info()
        # 4. int8 parttern fusion
        self._int_pattern_fusion()
        return self.model

    def _get_tensors_min_max(self):
        self._matmul_node_input_fp32()
        net = self.model.graph.dump_tensor()
        output_tensors = [content['input'] for content in \
            net['model']['operator'].values() if content['type'] == 'Output'][0]
        for output_tensor_name in output_tensors:
            self._tensors_min_max[output_tensor_name] = [float('inf'), float('-inf')]
        # get min/max of const tensors
        for node in self.model.nodes:
            weight_perm = node.attr['src1_perm'] if node.op_type == "InnerProduct" else None
            for input_tensor in node.input_tensors:
                if not input_tensor.source_op:
                    self._tensors_min_max[input_tensor.name] = [float('inf'), float('-inf')]
                    quantize_mode = 'per_tensor' if node.name not in self.config or \
                            self.config[node.name] =='fp32' else \
                            self.config[node.name]['weight']['granularity']
                    self._get_min_max(input_tensor, weight_perm, quantize_mode,
                                      self._tensors_min_max[input_tensor.name])

        self.model.graph.engine_init(net)
        for idx, (inputs, labels) in enumerate(self.dataloader):
            if idx > self.iterations:
                break
            else:
                # inference and put the tensors data to graph
                results = self.model.graph.inference(inputs)
                for tensor_name, tensor_data in results.items():
                    for node in self.model.nodes:
                        for input_tensor in node.input_tensors:
                            if tensor_name == input_tensor.name:
                                input_tensor.data = tensor_data
                                input_tensor.dtype = 'fp32'
                        for output_tensor in node.output_tensors:
                            if tensor_name == output_tensor.name:
                                output_tensor.data = tensor_data
                                output_tensor.dtype = 'fp32'
                
                # get min/max of output tensors
                for node in self.model.nodes:
                    for output_tensor in node.output_tensors:
                        if output_tensor.name in self._tensors_min_max and output_tensor.source_op:
                            quantize_mode = 'per_tensor' if node.name not in self.config or \
                                self.config[node.name] =='fp32' else\
                                self.config[node.name]['weight']['granularity']
                            self._get_min_max(output_tensor, None,
                                              quantize_mode,
                                              self._tensors_min_max[output_tensor.name])

    def _get_min_max(self, tensor, weight_perm, quantize_mode, tensor_min_max):
        tensor_name = tensor.name
        tensor_data = tensor.data
        tensor_min_history = tensor_min_max[0]
        tensor_max_history = tensor_min_max[1]
        # get min
        tensor_min_data = np.min(tensor_data)
        if not tensor.source_op and quantize_mode == "per_channel":
            if weight_perm == '0,1':
                tensor_min_data = np.min(tensor_data, axis=-1)
            else:
                tensor_min_data = np.min(tensor_data, axis=0)
        tensor_min_data = np.minimum(tensor_min_data, 0.).astype(np.float32)
        tensor_min_history = np.minimum(tensor_min_data, tensor_min_history).astype(np.float32)
        # get max
        tensor_max_data = np.max(tensor_data)
        if not tensor.source_op and quantize_mode == "per_channel":
            if weight_perm == '0,1':
                tensor_max_data = np.max(tensor_data, axis=-1)
            else:
                tensor_max_data = np.max(tensor_data, axis=0)
        tensor_max_data = np.maximum(tensor_max_data, 0.).astype(np.float32)
        tensor_max_history = np.maximum(tensor_max_data, tensor_max_history).astype(np.float32)
        self._tensors_min_max[tensor_name] = [np.array(tensor_min_history).astype(np.float32),
                                            np.array(tensor_max_history).astype(np.float32)]

    def _matmul_node_input_fp32(self):
        # if one of matmul pre nodes is fp32, both are fp32
        for node in self.model.nodes:
            if node.op_type == 'Matmul':
                is_fp32 = self._is_pre_matmul_fp32(node)
                if is_fp32:
                    self._another_pre_matmul_fp32(node)
                    self.config[node.name] ='fp32'

   # if pre node of matmul is in ordered to be fp32
    def _is_pre_matmul_fp32(self, node):
        pre_nodes_name = self.model.graph.get_pre_node_names(node.name)
        for pre_node_name in pre_nodes_name:
            pre_node_idx = self.model.get_node_id(pre_node_name)
            pre_node = self.model.nodes[pre_node_idx] 
            if pre_node.name in self.config and self.config[pre_node.name] =='fp32':
                return True
            elif pre_node.op_type in self._no_quantize_op:
                return self._is_pre_matmul_fp32(pre_node)
        return False

    # if another pre matmul quantized node is fp32
    def _another_pre_matmul_fp32(self, node):
        pre_nodes_name = self.model.graph.get_pre_node_names(node.name)
        for pre_node_name in pre_nodes_name:
            pre_node_idx = self.model.get_node_id(pre_node_name)
            pre_node = self.model.nodes[pre_node_idx]
            if pre_node.op_type in (self._quantize_op + self._part_quantize_op):
                self.config[pre_node.name] ='fp32'
            elif pre_node.op_type in self._no_quantize_op:
                self._another_pre_matmul_fp32(pre_node)

    def _insert_quantize_op(self):
        nodes = copy.deepcopy(self.model.nodes)
        for ip_node in nodes: 
            if ip_node.op_type == "InnerProduct" and self.config[ip_node.name] != 'fp32':
                ip_node_name = ip_node.name
                ip_node_idx = self.model.get_node_id(ip_node_name)
                ip_input_tensor = ip_node.input_tensors[0]
                pre_ip_node_name = ip_input_tensor.source_op[0]

                # pre ip_node
                pre_ip_node_idx = self.model.get_node_id(pre_ip_node_name)
                pre_ip_node = self.model.nodes[pre_ip_node_idx]
                ip_input_tensor = copy.deepcopy(ip_node.input_tensors[0])

                # 1. insert quantize op before innerproduct
                # change pre ip op's next op to quant op
                quant_node_name = ip_node_name + "_quant"
                # create quant output tensor 
                quant_output_tensor_name = ip_input_tensor.name + '_quant'
                if self.config[ip_node.name]['activation']['scheme'] == 'asym':
                    quant_output_tensor = Tensor(
                                            name = quant_output_tensor_name,
                                            source_op = [quant_node_name],
                                            dest_op = [ip_node.name],
                                            dtype='u8'
                                            )
                else:
                    quant_output_tensor = Tensor(
                                            name = quant_output_tensor_name,
                                            source_op = [quant_node_name],
                                            dest_op = [ip_node.name],
                                            dtype='s8'
                                            )                    
                self._tensors_min_max[quant_output_tensor_name] = \
                    self._tensors_min_max[ip_input_tensor.name]
                # create quant op
                quant_node = OPERATORS['QuantizeV2']()
                if self.config[ip_node.name]['activation']['scheme'] == 'asym':
                    quant_node.construct(name=quant_node_name, op_type='Quantize', 
                                        input_tensors=[ip_input_tensor],
                                        output_tensors=[quant_output_tensor],
                                        attr=OrderedDict({'output_dtype':'u8'}))
                else:
                    quant_node.construct(name=quant_node_name, op_type='Quantize', 
                                        input_tensors=[ip_input_tensor],
                                        output_tensors=[quant_output_tensor],
                                        attr=OrderedDict({'output_dtype':'s8'}))

                self.model.insert_nodes(ip_node_idx, [quant_node])
                # innerproduct input replace quantize output
                self.model.graph.change_node_input_tensors(ip_node_name, 0, quant_output_tensor)

        self._remove_duplicate_quantize_node()

    def _remove_duplicate_quantize_node(self):
        visit_tensors = []
        nodes = copy.deepcopy(self.model.nodes)
        for node in nodes:
            if node.op_type == "Quantize":
                input_name = node.input_tensors[0].name
                if input_name in visit_tensors:
                    self.model.remove_nodes([node.name])
                else:    
                    visit_tensors.append(input_name) 

    def _insert_quantize_info(self):
        for node in self.model.nodes:
            if (node.op_type in self._quantize_op and \
                self.config[node.name.split('_quant')[0]] != 'fp32') or \
                node.op_type in self._part_quantize_op:

                # insert input tensor min max
                node_type = node.op_type
                input_tensors = copy.deepcopy(node.input_tensors)
                input_size = len(input_tensors)

                # Don't quantize bias of inner_product and matmul
                input_size = 2 if input_size > 2 else input_size

                if node_type not in self._quantize_output_op:
                    for idx in range(input_size):
                        tensor = node.input_tensors[idx]
                        tensor_name = tensor.name
                        tensor_min_data = self._tensors_min_max[tensor_name][0]
                        tensor_min_shape = list(tensor_min_data.shape) \
                            if tensor_min_data.shape != () else [1]
                        tensor_min_name = tensor_name + '_min'
                        tensor_min = Tensor(name = tensor_min_name,
                                            dest_op = [node.name],
                                            data = tensor_min_data,
                                            shape = tensor_min_shape,
                                            dtype='fp32'
                                            )
                        node.input_tensors.append(tensor_min)

                        tensor_max_data = self._tensors_min_max[tensor_name][1]
                        tensor_max_shape = list(tensor_max_data.shape) \
                            if tensor_max_data.shape != () else [1]
                        tensor_max_name = tensor_name + '_max'
                        tensor_max = Tensor(name = tensor_max_name,
                                            dest_op = [node.name],
                                            data = tensor_max_data,
                                            shape = tensor_max_shape,
                                            dtype='fp32'
                                            )
                        node.input_tensors.append(tensor_max)

                # 2. insert output tensor min max
                if node_type not in self._quantize_input_op:
                    output_tensors = copy.deepcopy(node.output_tensors)
                    for idx in range(len(output_tensors)):
                        tensor = node.output_tensors[idx]
                        tensor_name = tensor.name
                        tensor_data = self._tensors_min_max[tensor_name]

                        tensor_min_data = self._tensors_min_max[tensor_name][0]
                        tensor_min_shape = list(tensor_min_data.shape) \
                            if tensor_min_data.shape != () else [1]
                        tensor_min_name = tensor_name + '_min'
                        tensor_min = Tensor(name = tensor_min_name,
                                            dest_op = [node.name],
                                            data = tensor_min_data,
                                            shape = tensor_min_shape,
                                            dtype='fp32'
                                            )
                        node.input_tensors.append(tensor_min)

                        tensor_max_data = self._tensors_min_max[tensor_name][1]
                        tensor_max_shape = list(tensor_max_data.shape) \
                            if tensor_max_data.shape != () else [1]
                        tensor_max_name = tensor_name + '_max'
                        tensor_max = Tensor(name = tensor_max_name,
                                            dest_op = [node.name],
                                            data = tensor_max_data,
                                            shape = tensor_max_shape,
                                            dtype='fp32'
                                            )
                        node.input_tensors.append(tensor_max)

    def _quantize_innerproduct(self):
        for node in self.model.nodes:
            if node.op_type == "InnerProduct" and \
                self.config[node.name.split('_quant')[0]] != 'fp32':

                input_tensor = node.input_tensors[0]
                weight_tensor = node.input_tensors[1]
                weight_perm = node.attr['src1_perm'] 
                
                self._quant_weight(weight_tensor, weight_perm)

                bias_tensor = node.input_tensors[2] if len(node.input_tensors) > 2 else None
                if (bias_tensor and not bias_tensor.source_op) or not bias_tensor:
                    self._quant_bias(node, input_tensor, weight_tensor, bias_tensor, weight_perm)

    def _get_scale_zp(self, tensor, dtype):
        tensor_name = tensor.name
        tensor_min_data = self._tensors_min_max[tensor_name][0]
        tensor_max_data = self._tensors_min_max[tensor_name][1]

        if (dtype=='u8'):
            max_sub_min = tensor_max_data - tensor_min_data
            scale = np.where(max_sub_min==0., 0., 255./max_sub_min)
            zero_point = -tensor_min_data * scale
        elif (dtype=='s8'):
            max_abs = np.maximum(np.fabs(tensor_min_data), np.fabs(tensor_max_data))
            scale = np.where(max_abs==0., 0., 127./max_abs)
            zero_point = 0.

        return scale, zero_point

    def _quant_weight(self, weight_tensor, weight_perm):
        weight_data = weight_tensor.data
        weight_scale , weight_zero_point = self._get_scale_zp(weight_tensor, 's8')

        if weight_perm=="0,1":
            weight_tensor.data = \
                (np.round((weight_tensor.data).T * weight_scale).astype(np.int8)).T
        else:
            weight_tensor.data = np.round(weight_tensor.data * weight_scale).astype(np.int8)
        weight_tensor.dtype = 's8'

    def _quant_bias(self, node, input_tensor, weight_tensor, bias_tensor, weight_perm):
        input_data_min = self._tensors_min_max[input_tensor.name][0]
        weight_s8 = weight_tensor.data
        if self.config[node.name]['activation']['scheme'] == 'asym':
            input_scale , input_zero_point = self._get_scale_zp(input_tensor, 'u8')
        else:
            input_scale , input_zero_point = self._get_scale_zp(input_tensor, 's8')
        weight_scale , weight_zero_point = self._get_scale_zp(weight_tensor, 's8')

        if self.config[node.name]['activation']['scheme'] == 'asym':
            if weight_perm == '0,1':
                bias_zero_point = input_scale * input_data_min * \
                    np.sum(weight_s8.astype(float), axis=-1)
            else:
                bias_zero_point = input_scale * input_data_min * \
                    np.sum(weight_s8.astype(float), axis=0)
            bias_s32 = bias_tensor.data * input_scale * weight_scale if bias_tensor else 0.
            bias_s32 = np.round(bias_s32 + bias_zero_point).astype(np.int32)
            if not bias_tensor:
                bias_tensor_name = weight_tensor.name + "_bias"
                bias_shape = list(bias_zero_point.shape) \
                    if bias_zero_point.shape != () else [1]
                bias_tensor = Tensor(name = bias_tensor_name,
                                dest_op = [node.name],
                                data = bias_s32,
                                shape = bias_shape,
                                dtype='s32'
                                )
                node.input_tensors.append(bias_tensor)
            else:
                bias_tensor.data = bias_s32
                bias_tensor.dtype = 's32'
        else:
            if bias_tensor:
                bias_s32 = np.round(bias_tensor.data * input_scale * weight_scale).astype(np.int32)
                bias_tensor.data = bias_s32
                bias_tensor.dtype = 's32'

    def _int_pattern_fusion(self):
        self._quantize_node_out_int8()
        self._quantize_fusion()

    # quant op output int8
    def _quantize_node_out_int8(self):
        for node in self.model.nodes:
            if (node.op_type in self._quantize_op and \
                    self.config[node.name.split('_quant')[0]] != 'fp32') or \
                    node.op_type in self._quantize_output_op:
                is_output_int8 = True 
                is_output_int8 = self._is_output_int8(node, is_output_int8)
                if is_output_int8:
                    if node.op_type == "Softmax":
                        node.output_tensors[0].dtype = 'u8'
                        node.attr['output_dtype'] = 'u8'
                    else:
                        node.output_tensors[0].dtype = 's8'
                        node.attr['output_dtype'] = 's8'

    # if quant op can output int8
    def _is_output_int8(self, node, is_output_int8):
        for output in node.output_tensors:
            for next_node_name in output.dest_op:
                next_node_idx = self.model.get_node_id(next_node_name)
                next_node = self.model.nodes[next_node_idx]
                if next_node.op_type in self._quantize_op and \
                        self.config[next_node.name.split('_quant')[0]] != 'fp32':
                    return is_output_int8 & True
                elif next_node.op_type in self._no_quantize_op:
                    return self._is_output_int8(next_node, is_output_int8)
                else:
                    return is_output_int8 & False

    # fuse quantize_output_op with quantize
    def _quantize_fusion(self):
        for node in self.model.nodes:
            if (node.op_type in self._quantize_op and \
                    self.config[node.name.split('_quant')[0]] != 'fp32') or \
                    node.op_type in self._quantize_output_op:
                is_quantize_fusion = True
                is_quantize_fusion, pre_quantize_node, quantize_node = self._is_quantize_fusion(
                    node, is_quantize_fusion)
                if is_quantize_fusion:
                    # if node.op_type != 'InnerProduct':
                    self.model.remove_nodes([quantize_node.name])
                    pre_quantize_node.output_tensors[0] = quantize_node.output_tensors[0]
                    if self.config[node.name]['activation']['scheme'] == 'asym':
                        node.attr['output_dtype'] = 'u8'
                    else:
                        node.attr['output_dtype'] = 's8'

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
                else:
                    is_qunatize_fusion &= False
                    return is_qunatize_fusion & False, None, None

        is_qunatize_fusion &= False
        return is_qunatize_fusion, None, None
