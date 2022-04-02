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
import onnx
import logging
from onnx import onnx_pb as onnx_proto
from onnx import TensorProto

from onnxruntime.quantization.quant_utils import QuantizedValueType
from onnxruntime.quantization.quant_utils import find_by_name
from onnxruntime.quantization.quant_utils import __producer__, __version__

from neural_compressor.adaptor.ox_utils.registry import CreateQDQQuantizer
from neural_compressor.adaptor.ox_utils.util import QuantizedValue
from neural_compressor.utils import OPTIONS
from neural_compressor.adaptor.ox_utils.onnx_quantizer import ONNXQuantizer

logger = logging.getLogger()

class QDQQuantizer(ONNXQuantizer):
    def __init__(self, model, q_config, mode, static, quantization_params,
                 op_types_to_quantize):
        super().__init__(model, q_config, mode, static, quantization_params, 
                 op_types_to_quantize)

        model = onnx.shape_inference.infer_shapes(model)
        self.value_infos = {vi.name: vi for vi in model.graph.value_info}
        self.value_infos.update({ot.name: ot for ot in model.graph.output})
        self.value_infos.update({it.name: it for it in model.graph.input})
        self.option = OPTIONS['onnxrt_qdqops']
        self.replace_output = []
        self.replace_input= []
        self.remove_nodes = []

        self.op_types_to_exclude_output_quantization = [] \
            if not self.option.qdq_setting.OpTypesToExcludeOutputQuantizatioin \
            else self.option.qdq_setting.OpTypesToExcludeOutputQuantizatioin

        self.add_qdq_pair_to_weight = self.option.qdq_setting.AddQDQPairToWeight

        self.dedicated_qdq_pair = self.option.qdq_setting.DedicatedQDQPair
        if self.dedicated_qdq_pair:
            self.tensor_to_its_receiving_nodes = {}

    def should_quantize(self, node):
        if node.name in self.config:
            return self.config[node.name] != 'fp32' and \
                any([i in self.quantization_params for i in node.input])
        else:
            return False

    def quantize_model(self):
        if self.opset_version < 13:
            logger.warning("Per-Channel support with QDQ format requires onnx opset >= 13," \
                " use per-tensor granularity instead")
 
        if self.dedicated_qdq_pair:
            for node in self.model.nodes():
                if self.should_quantize(node):
                    for tensor_name in node.input:
                        if tensor_name not in self.tensor_to_its_receiving_nodes:
                            self.tensor_to_its_receiving_nodes[tensor_name] = []
                        self.tensor_to_its_receiving_nodes[tensor_name].append(node)

        for node in self.model.nodes():
            if self.should_quantize(node):
                op_quantizer = CreateQDQQuantizer(self, node)
                op_quantizer.quantize()

        self.model.graph().node.extend(self.new_nodes)
        for item in self.replace_output:
            self.model.replace_output_of_all_nodes(item[0], item[1], ['DequantizeLinear'])
        for item in self.replace_input:
            self.model.replace_input_of_all_nodes(item[0], item[1], ['QuantizeLinear'])

        for node in self.remove_nodes:
            self.model.remove_node(node)

        self.model.remove_unused_constant()

        self.model.model.producer_name = __producer__
        self.model.model.producer_version = __version__

        return self.model.model

    def quantize_tensor(self, tensor_name, initializer_use_weight_qType=True):
        if tensor_name in self.quantized_value_map.keys():
            return

        # Quantize the input
        initializer = find_by_name(tensor_name, self.model.initializer())
        if initializer is not None:
            if initializer.data_type != onnx_proto.TensorProto.FLOAT:
                return
            node = self.model.input_name_to_nodes[tensor_name][0]
            if node.op_type not in self.op_types_to_quantize:
                dtype = onnx_proto.TensorProto.INT8 if initializer_use_weight_qType \
                    else onnx_proto.TensorProto.UINT8
                scheme = 'sym' if initializer_use_weight_qType else 'asym'
            else:
                dtype = self.config[node.name]['weight']['dtype'] if \
                    initializer_use_weight_qType else \
                    self.config[node.name]['activation']['dtype']
                scheme = self.config[node.name]['weight']['scheme'] if \
                    initializer_use_weight_qType else \
                    self.config[node.name]['activation']['scheme']
            if self.add_qdq_pair_to_weight:
                weight = self._get_quantized_weight(initializer, dtype, scheme)
                self._update_weight(weight)
                q_weight_name = weight.name + "_quantized"
                zp_name = weight.name + "_zero_point"
                scale_name = weight.name + "_scale"
                                        
                qlinear_node = onnx.helper.make_node("QuantizeLinear", 
                                                     [tensor_name, scale_name, zp_name],
                                                     [tensor_name + "_QuantizeLinear"],
                                                     tensor_name + "_QuantizeLinear")
                dequant_node = onnx.helper.make_node("DequantizeLinear",
                                        [tensor_name + "_QuantizeLinear", scale_name, zp_name],
                                        [tensor_name + "_DequantizeLinear"],
                                        tensor_name + "_DequantizeLinear")
                self.replace_input.append([tensor_name, tensor_name + "_DequantizeLinear"])

                self.new_nodes.extend([qlinear_node, dequant_node])
            else:
                weight = self._get_quantized_weight(initializer, dtype, scheme)
                self._update_weight(weight)
                q_weight_name = weight.name + "_quantized"
                zp_name = weight.name + "_zero_point"
                scale_name = weight.name + "_scale"
 
                inputs = [q_weight_name, scale_name, zp_name]
                output_name = tensor_name + '_DequantizeLinear'
                node = onnx.helper.make_node("DequantizeLinear", inputs, [output_name],
                                             tensor_name + '_DequantizeLinear')
                self.new_nodes.append(node)
                self.replace_input.append([tensor_name, tensor_name + "_DequantizeLinear"])
        else:
            if tensor_name in self.value_infos and \
                self.value_infos[tensor_name].type.HasField('tensor_type') and \
                self.value_infos[tensor_name].type.tensor_type.elem_type != TensorProto.FLOAT:
                return
            data_found, scale_name, zp_name, _, _ = self._get_quantization_params(tensor_name)

            if data_found == False:
                raise ValueError(
                    "Quantization parameters are not specified for param {}."
                    "In static mode quantization params for inputs and outputs \
                    of nodes to be quantized are required.".format(tensor_name))

            if self.dedicated_qdq_pair and \
                tensor_name in self.tensor_to_its_receiving_nodes and \
                len(self.tensor_to_its_receiving_nodes[tensor_name]) > 1:
                num_dedicated_qdq_pair = len(self.tensor_to_its_receiving_nodes[tensor_name])
                for i in range(num_dedicated_qdq_pair):
                    postfix = str(i+1)
                    q_input = tensor_name
                    q_output = tensor_name + "_QuantizeLinear_" + postfix 
                    dq_input = q_output
                    dq_output = tensor_name + "_DequantizeLinear_" + postfix
                    quant_node_name = tensor_name + "_QuantizeLinear_" + postfix
                    dequant_node_name = tensor_name + "_DequantizeLinear_" + postfix
                    qlinear_node = onnx.helper.make_node("QuantizeLinear", 
                                                         [q_input, scale_name, zp_name],
                                                         [q_output], quant_node_name)
                    dequant_node = onnx.helper.make_node("DequantizeLinear",
                                                         [dq_input, scale_name, zp_name],
                                                         [dq_output],
                                                         dequant_node_name)
                    self.new_nodes.extend([qlinear_node, dequant_node])

                    quantized_value = QuantizedValue(tensor_name, dq_output, scale_name, zp_name,
                                                     QuantizedValueType.Input)
                    self.quantized_value_map[tensor_name] = quantized_value
            else:
                q_input = tensor_name
                q_output = tensor_name + "_QuantizeLinear"
                dq_input = q_output
                dq_output = tensor_name + "_DequantizeLinear"
                if tensor_name in [i.name for i in self.model.model.graph.output]:
                    q_input = tensor_name + "_QuantizeLinearInput"
                    dq_output = tensor_name
                    self.replace_output.append([tensor_name, q_input])
                else:
                    self.replace_input.append([tensor_name, dq_output])

                quant_node_name = tensor_name + "_QuantizeLinear"
                dequant_node_name = tensor_name + "_DequantizeLinear"
                qlinear_node = onnx.helper.make_node("QuantizeLinear", 
                                                     [q_input, scale_name, zp_name],
                                                     [q_output], quant_node_name)
                dequant_node = onnx.helper.make_node("DequantizeLinear",
                                                     [dq_input, scale_name, zp_name],
                                                     [dq_output],
                                                     dequant_node_name)
                self.new_nodes.extend([qlinear_node, dequant_node])

                quantized_value = QuantizedValue(tensor_name, dq_output, scale_name, zp_name,
                                                 QuantizedValueType.Input)
                self.quantized_value_map[tensor_name] = quantized_value

    def quantize_bias_tensor(self, bias_name, input_name, weight_name):
        if bias_name in self.quantized_value_map.keys():
            return
        # Quantize the input
        self.quantize_bias(bias_name, input_name, weight_name)
        self.model.remove_initializer(find_by_name(bias_name, self.model.initializer()))
        quant_value = self.quantized_value_map[bias_name]
        inputs = [quant_value.q_name, quant_value.scale_name, quant_value.zp_name]
        if quant_value.axis is not None:
            dequant_node = onnx.helper.make_node("DequantizeLinear",
                                                 inputs, [bias_name],
                                                 bias_name + '_DequantizeLinear',
                                                 axis=quant_value.axis)
        else:
            dequant_node = onnx.helper.make_node("DequantizeLinear", inputs, [bias_name],
                                                 bias_name + '_DequantizeLinear')
        self.new_nodes.append(dequant_node)

    def quantize_weights_per_channel(self, weight_name, weight_qType, scheme, axis):
        if self.opset_version < 13:
            self.quantize_tensor(weight_name)
            return

        if self.add_qdq_pair_to_weight:
            q_name, zp_name, scale_name = self.quantize_weight_per_channel(weight_name, 
                                                                           weight_qType,
                                                                           scheme,
                                                                           axis) 
            qlinear_node = onnx.helper.make_node("QuantizeLinear", 
                                                 [weight_name, scale_name, zp_name],
                                                 [weight_name + "_QuantizeLinear"],
                                                 weight_name + "_QuantizeLinear",
                                                 axis=axis)
            dequant_node = onnx.helper.make_node("DequantizeLinear",
                                        [weight_name + "_QuantizeLinear", scale_name, zp_name],
                                        [weight_name + "_DequantizeLinear"],
                                        weight_name + "_DequantizeLinear",
                                        axis=axis)
            self.replace_input.append([weight_name, weight_name + "_DequantizeLinear"])

            self.new_nodes.extend([qlinear_node, dequant_node])
        else:
            q_name, zp_name, scale_name = self.quantize_weight_per_channel(weight_name, 
                                                                           weight_qType,
                                                                           scheme,
                                                                           axis)

            inputs = [q_name, scale_name, zp_name]
            output_name = weight_name + "_DequantizeLinear"
            node = onnx.helper.make_node("DequantizeLinear",
                                         inputs, [output_name],
                                         weight_name + '_DequantizeLinear',
                                         axis=axis)
            self.new_nodes.append(node)

            # Replace weight_name with output of DequantizeLinear
            self.replace_input.append([weight_name, output_name])
