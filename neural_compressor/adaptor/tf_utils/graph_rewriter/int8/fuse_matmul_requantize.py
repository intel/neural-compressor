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

import numpy as np
import tensorflow as tf

from tensorflow.python.framework import tensor_util
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import dtypes

from ..graph_base import GraphRewriterBase
from ..graph_util import GraphAnalyzer
from ..graph_util import GraphRewriterHelper as Helper


class FuseMatMulRequantizeDequantizeTransformer(GraphRewriterBase):
    """Fuse QuantizedMatMul + Requantize + Dequantize into QuantizedMatMulWithBiasAndDequantize.
    """
    fuse_patterns = {
        "2.7.0": [["QuantizedMatMulWithBias"], ['Requantize'], ['Dequantize'],
                  ('Softmax',)],
        "2.6.0": [["QuantizedMatMulWithBias"], ['Requantize'], ['Dequantize'],
                  ('Softmax',)],
        "2.5.0": [["QuantizedMatMulWithBias"], ['Requantize'], ['Dequantize'],
                  ('Softmax',)],
        "2.4.0": [["QuantizedMatMulWithBias"], ['Requantize'], ['Dequantize'],
                  ('Softmax',)],
        "2.3.0": [["QuantizedMatMulWithBias"], ['Requantize'], ['Dequantize'],
                  ('Softmax',)],
        "2.2.0": [["QuantizedMatMulWithBias"], ['Requantize'], ['Dequantize'],
                   ('Softmax',)],
        "1.15.0-up3":  [["QuantizedMatMulWithBias"], ['Requantize'], ['Dequantize'],
                        ('Softmax',)],
        "1.15.0-up2":  [["QuantizedMatMulWithBias"], ['Requantize'], ['Dequantize'],
                        ('Softmax',)],
        "default": []}

    def __init__(self, model, device='cpu'):
        super().__init__(model)
        self.device = device
        self.graph_analyzer = GraphAnalyzer()
        self.graph_analyzer.graph = self.model

        self.graph_info = self.graph_analyzer.parse_graph()
        self.version = tf.version.VERSION if tf.version.VERSION in self.fuse_patterns \
             else 'default'
        self.eps = 1e-5

    def do_transformation(self):
        float32_type = dtypes.float32.as_datatype_enum
        qint32_type = dtypes.qint32.as_datatype_enum
        target_nodes = self.graph_analyzer.query_fusion_pattern_nodes(
                self.fuse_patterns[self.version])
        for i in target_nodes:
            # TODO Remove below checker once the TF's limitation removed.
            if len(i) == 5:
                continue

            quantized_node_name = i[0]
            quantized_node = self.graph_info[quantized_node_name].node
            requantize_node_name = i[1]
            requantize_node = self.graph_info[requantize_node_name].node
            requested_output_min_name = requantize_node.input[3]
            requested_output_max_name = requantize_node.input[4]
            deq_node_name = i[2]

            quantized_node_op = i[-1][0]

            new_node = node_def_pb2.NodeDef()

            new_node.op = quantized_node_op + "AndDequantize"
            new_node.name = requantize_node_name
            for _, value in enumerate(quantized_node.input):
                new_node.input.append(value)

            new_node.input.append(requested_output_min_name)
            new_node.input.append(requested_output_max_name)
            if 'T1' in quantized_node.attr:
                new_node.attr["T1"].CopyFrom(quantized_node.attr['T1'])
            if 'T2' in quantized_node.attr:
                new_node.attr["T2"].CopyFrom(quantized_node.attr['T2'])

            top_node_name = Helper.node_name_from_input(quantized_node.input[0])
            max_filter_node = self.graph_info[new_node.input[6]].node
            min_filter_node = self.graph_info[new_node.input[5]].node
            last_node = self.graph_info[new_node.input[0]].node

            bias_node = self.graph_info[Helper.node_name_from_input(new_node.input[2])].node
            max_input_node = self.graph_info[last_node.input[-1]].node
            min_input_node = self.graph_info[last_node.input[-2]].node

            if max_filter_node.op == 'Const':
                min_input_value = (min_input_node.attr['value'].tensor.float_val)[0]
                max_input_value = (max_input_node.attr['value'].tensor.float_val)[0]

                max_filter_value = (max_filter_node.attr['value'].tensor.float_val)[0]
                min_filter_value = (min_filter_node.attr['value'].tensor.float_val)[0]

                weights_tensor = tensor_util.MakeNdarray(
                        self.graph_info[new_node.input[1]].node.attr['value'].tensor)
                bias_tensor = tensor_util.MakeNdarray(
                    self.graph_info[new_node.input[2]].node.attr['value'].tensor)
                is_min_first = bool(quantized_node.attr['input_quant_mode'].s == b'MIN_FIRST')
                input_range = max_input_value - min_input_value if is_min_first else max(
                        abs(max_input_value), abs(min_input_value))

                if  -self.eps <= input_range <= self.eps:
                    input_range += self.eps

                if -self.eps <= max_input_value - min_input_value <= self.eps:
                    max_input_value += self.eps
                int32_bias = Helper.generate_int32_bias_for_matmul(bias_tensor, weights_tensor,
                                                        input_range, max_input_value,
                                                        min_input_value,
                                                        max_filter_value, min_filter_value)

                bias_node.attr['dtype'].CopyFrom(
                    attr_value_pb2.AttrValue(
                        type=float32_type if self.device == 'gpu' else qint32_type))
                bias_node.attr['value'].CopyFrom(
                    attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
                        bias_tensor if self.device == 'gpu' else int32_bias, dtypes.
                        float32 if self.device == 'gpu' else dtypes.int32, bias_tensor.shape)))

                bias_node.attr['value'].tensor.dtype = float32_type \
                                        if self.device == 'gpu' else qint32_type
                new_node.attr["Tbias"].CopyFrom(attr_value_pb2.AttrValue(type=float32_type \
                                                if self.device == 'gpu' else qint32_type))
            else:
                new_node.attr["Tbias"].CopyFrom(attr_value_pb2.AttrValue(type=float32_type))
            new_node.attr["Toutput"].CopyFrom(attr_value_pb2.AttrValue(type=float32_type))

            self.graph_analyzer.remove_node(requantize_node_name)

            if self.graph_info[deq_node_name].outputs:
                self.graph_analyzer.replace_single_node(
                    new_node, [top_node_name], quantized_node_name,
                    self.graph_info[deq_node_name].outputs, deq_node_name)
                self.graph_analyzer.remove_node(deq_node_name)
            else:
                self.graph_analyzer.remove_node(deq_node_name)

                new_node.name = deq_node_name
                self.graph_analyzer.replace_single_node(
                    new_node, [top_node_name], quantized_node_name, [], deq_node_name)

            self.graph_analyzer.remove_node(quantized_node_name)

        return self.graph_analyzer.dump_graph()
class FuseMatMulRequantizeTransformer(GraphRewriterBase):
    """Fuse Quantized MatMul Op with the successor Requantize Op.
    """
    fuse_patterns = {"2.2.0": [["QuantizedMatMulWithBiasAndRelu"], ['Requantize']],
                     "2.1.0": [["QuantizedMatMulWithBiasAndRelu"], ['Requantize']],
                     "default": [["QuantizedMatMulWithBiasAndRelu"], ['Requantize']]}

    def __init__(self, model, device='cpu'):
        super().__init__(model)
        self.device = device
        self.graph_analyzer = GraphAnalyzer()
        self.graph_analyzer.graph = self.model
        self.eps = 1e-05
        self.graph_info = self.graph_analyzer.parse_graph()

    def do_transformation(self):
        """Fuse the quantized op with the following requantize op.
        Returns:
            [graphdef]: the optimized graphdef object
        """
        uint8_type = dtypes.quint8.as_datatype_enum
        float32_type = dtypes.float32.as_datatype_enum
        qint32_type = dtypes.qint32.as_datatype_enum

        while True:
            target_nodes = self.graph_analyzer.query_fusion_pattern_nodes(
                self.fuse_patterns['default'])
            if len(target_nodes) == 0:
                break

            i = target_nodes[0]
            quantized_node_name = i[0]
            quantized_node = self.graph_info[quantized_node_name].node
            requantize_node_name = i[1]
            requantize_node = self.graph_info[requantize_node_name].node
            requested_output_min_name = requantize_node.input[3]
            requested_output_max_name = requantize_node.input[4]

            quantized_node_op = i[-1][0]

            new_node = node_def_pb2.NodeDef()

            new_node.op = quantized_node_op + "AndRequantize"
            new_node.name = requantize_node_name
            for _, value in enumerate(quantized_node.input):
                new_node.input.append(value)
            new_node.input.append(requested_output_min_name)
            new_node.input.append(requested_output_max_name)
            if 'T1' in quantized_node.attr:
                new_node.attr["T1"].CopyFrom(quantized_node.attr['T1'])
            if 'T2' in quantized_node.attr:
                new_node.attr["T2"].CopyFrom(quantized_node.attr['T2'])

            parent_node_name = Helper.node_name_from_input(quantized_node.input[0])
            max_filter_node = self.graph_info[new_node.input[6]].node
            min_filter_node = self.graph_info[new_node.input[5]].node
            last_node = self.graph_info[new_node.input[0]].node

            is_min_first = bool(quantized_node.attr['input_quant_mode'].s == b'MIN_FIRST')

            bias_node = self.graph_info[new_node.input[2]].node
            max_input_node = self.graph_info[last_node.input[-1]].node
            min_input_node = self.graph_info[last_node.input[-2]].node

            if last_node.op.find('Requantize') != -1 or last_node.op.find('QuantizeV2') != -1:
                min_input_value = (min_input_node.attr['value'].tensor.float_val)[0]
                max_input_value = (max_input_node.attr['value'].tensor.float_val)[0]

                max_filter_value = (max_filter_node.attr['value'].tensor.float_val)[0]
                min_filter_value = (min_filter_node.attr['value'].tensor.float_val)[0]

                weights_tensor = tensor_util.MakeNdarray(
                        self.graph_info[new_node.input[1]].node.attr['value'].tensor)
                bias_tensor = tensor_util.MakeNdarray(
                    self.graph_info[new_node.input[2]].node.attr['value'].tensor)

                input_range = max_input_value - min_input_value if is_min_first else max(
                        abs(max_input_value), abs(min_input_value))

                if -self.eps <= input_range <= self.eps:
                    input_range += self.eps

                if -self.eps <= max_input_value - min_input_value <= self.eps:
                    max_input_value += self.eps
                int32_bias = Helper.generate_int32_bias_for_matmul(bias_tensor, weights_tensor,
                                                                   input_range,
                                                                   max_input_value,
                                                                   min_input_value,
                                                                   max_filter_value,
                                                                   min_filter_value)
                bias_node.attr['dtype'].CopyFrom(
                    attr_value_pb2.AttrValue(
                        type=float32_type if self.device == 'gpu' else qint32_type))
                bias_node.attr['value'].CopyFrom(
                    attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
                        bias_tensor if self.device == 'gpu' else int32_bias, dtypes.
                        float32 if self.device == 'gpu' else dtypes.int32, bias_tensor.shape)))

                bias_node.attr['value'].tensor.dtype = float32_type \
                                        if self.device == 'gpu' else qint32_type
                new_node.attr["Tbias"].CopyFrom(attr_value_pb2.AttrValue(type=float32_type \
                                                if self.device == 'gpu' else qint32_type))

                new_node.attr["Toutput"].CopyFrom(
                        attr_value_pb2.AttrValue(type=uint8_type))
                #TODO enabled below commit once the graph refactor pre_optimize commmitted.
                if quantized_node_op.find('Relu') == -1:
                    deq_node_name = self.graph_info[requantize_node_name].outputs[0]
                    deq_node = self.graph_info[deq_node_name].node
                    deq_node.attr['T'].CopyFrom(attr_value_pb2.AttrValue(type=uint8_type))
            else:
                new_node.attr["Tbias"].CopyFrom(attr_value_pb2.AttrValue(type=float32_type))

            self.graph_analyzer.replace_single_node(
                new_node, [parent_node_name], quantized_node_name,
                [self.graph_info[requantize_node_name].outputs[0]], requantize_node_name)
            self.graph_analyzer.remove_node(quantized_node_name)

        return self.graph_analyzer.dump_graph()
