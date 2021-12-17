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

class EngineBf16Quantizer(engine_quantizer.EngineQuantizer):

    def __init__(self, model, dataloader, iterations, q_config, op_types_to_quantize):
        super(EngineBf16Quantizer, self).__init__(
            model, dataloader, iterations, q_config, op_types_to_quantize)
        self._bf16_weight = ["InnerProduct"]

    # 1. quant node weight
    def _quant_node(self):
        self._copy_const()
        for node in self.model.nodes:
            if node.op_type in self._bf16_weight and self.config[node.name] != 'fp32':
                weight_tensor = node.input_tensors[1]
                weight_tensor.data = self._fp32_to_bf16(weight_tensor.data)
                weight_tensor.dtype = 'bf16'
                if len(node.input_tensors) > 2:
                    bias_tensor = node.input_tensors[2]
                    bias_tensor.data = self._fp32_to_bf16(bias_tensor.data)
                    bias_tensor.dtype = 'bf16'

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
                        quantize_op_output_tensor = Tensor(
                                                name = quantize_op_output_name,
                                                source_op = [quantize_op_name],
                                                dest_op = [quant_node.name],
                                                dtype='bf16'
                                                )

                        # create quantize op
                        quantize_op = OPERATORS['QuantizeV2']()
                        quantize_op.construct(name=quantize_op_name, op_type='Quantize', 
                                            input_tensors=[quant_input_tensor],
                                            output_tensors=[quantize_op_output_tensor],
                                            attr=OrderedDict({'output_dtype':'bf16'}))

                        quant_node_idx = self.model.get_node_id(quant_node.name)
                        self.model.insert_nodes(quant_node_idx, [quantize_op])
                        # replace quant's input with quantize op output
                        self.model.graph.change_node_input_tensors(quant_node.name, \
                                                                idx, quantize_op_output_tensor)


        self._remove_duplicate_quantize_op()


    def _fp32_to_bf16(self, fp32_np):
        assert(fp32_np.dtype==np.float32)
        int32_np = fp32_np.view(dtype=np.int32)
        int32_np = int32_np >> 16
        bf16_np = int32_np.astype(np.uint16)
        return bf16_np

    # def _bf16_to_fp32(self, bf16_np):
    #     assert(bf16_np.dtype==np.uint16)
    #     int32_np = bf16_np.view(np.int32)
    #     int32_np = int32_np << 16
    #     fp32_np = int32_np.astype(np.float32)
    #     return fp32_np
