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
import numpy as np
import copy
from collections import namedtuple
from collections import OrderedDict
from engine.compile.ops.op import OPERATORS, Tensor
import neural_compressor.adaptor.engine_utils.engine_quantizer as engine_quantizer

class EngineInt8Quantizer(engine_quantizer.EngineQuantizer):

    def __init__(self, model, dataloader, iterations, q_config, op_types_to_quantize):
        super(EngineInt8Quantizer, self).__init__(
            model, dataloader, iterations, q_config, op_types_to_quantize)
        self._tensors_min_max = {}
        self._quantize_input_op = ["Quantize"]
        self._quantize_output_op = ["Softmax"]
        self._part_quantize_op = self._quantize_input_op + self._quantize_output_op

    # 0. get min/max of tensor
    def _get_tensors_min_max(self):
        self._copy_const()
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
                                if input_tensor.dtype == None:
                                    input_tensor.dtype = 'fp32'
                        for output_tensor in node.output_tensors:
                            if tensor_name == output_tensor.name:
                                output_tensor.data = tensor_data
                                if output_tensor.dtype == None:
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

    # 1. quant node weight
    def _quant_node(self):
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

    # 2. insert quantize before quant node
    def _insert_quantize_op(self):
        nodes = copy.deepcopy(self.model.nodes)
        for quant_node in nodes:
            if quant_node.op_type in self._quantize_op and \
                self.config[quant_node.name] != 'fp32':
                for idx, quant_input_tensor in enumerate(quant_node.input_tensors):
                    if quant_input_tensor.source_op:
                        if 'append_op' in quant_node.attr and \
                            (quant_node.attr['append_op'] == 'sum' \
                            or quant_node.attr['append_op'] == 'binary_add' ) \
                            and idx == len(quant_node.input_tensors) - 1:
                            continue
                        # create output tensor of quantize op
                        pre_quant_node_name = quant_input_tensor.source_op[0]
                        pre_quant_node_idx = self.model.get_node_id(pre_quant_node_name)
                        pre_quant_node = self.model.nodes[pre_quant_node_idx]
                        quantize_op_name = quant_node.name + "_quant_" + str(idx)
                        quantize_op_output_name = quant_input_tensor.name + '_quant'
                        if self.config[quant_node.name]['activation']['scheme'] == 'asym' or \
                            pre_quant_node.op_type == 'Softmax':
                            quantize_op_output_tensor = Tensor(
                                                    name = quantize_op_output_name,
                                                    source_op = [quantize_op_name],
                                                    dest_op = [quant_node.name],
                                                    dtype='u8'
                                                    )
                        elif self.config[quant_node.name]['activation']['scheme'] == 'sym':
                            quantize_op_output_tensor = Tensor(
                                                    name = quantize_op_output_name,
                                                    source_op = [quantize_op_name],
                                                    dest_op = [quant_node.name],
                                                    dtype='s8'
                                                    )
                        else:
                            assert self.config[quant_node.name]['activation']['scheme'] == 'asym'\
                                or self.config[quant_node.name]['activation']['scheme'] == 'sym' 

                        # create quantize op
                        quantize_op = OPERATORS['QuantizeV2']()
                        if self.config[quant_node.name]['activation']['scheme'] == 'asym' or \
                            pre_quant_node.op_type == 'Softmax':
                            quantize_op.construct(name=quantize_op_name, op_type='Quantize', 
                                                input_tensors=[quant_input_tensor],
                                                output_tensors=[quantize_op_output_tensor],
                                                attr=OrderedDict({'output_dtype':'u8'}))
                        elif self.config[quant_node.name]['activation']['scheme'] == 'sym':
                            quantize_op.construct(name=quantize_op_name, op_type='Quantize', 
                                                input_tensors=[quant_input_tensor],
                                                output_tensors=[quantize_op_output_tensor],
                                                attr=OrderedDict({'output_dtype':'s8'}))
                        else:
                            assert self.config[quant_node.name]['activation']['scheme'] == 'asym'\
                                or self.config[quant_node.name]['activation']['scheme'] == 'sym' 

                        quant_node_idx = self.model.get_node_id(quant_node.name)
                        self.model.insert_nodes(quant_node_idx, [quantize_op])
                        # replace quant's input with quantize op output
                        self.model.graph.change_node_input_tensors(
                            quant_node.name, idx, quantize_op_output_tensor)

                        # updata tensor min/max
                        self._tensors_min_max[quantize_op_output_name] = \
                            self._tensors_min_max[quant_input_tensor.name]

        self._remove_duplicate_quantize_op()


    # 4. insert quant info to each tensor
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
