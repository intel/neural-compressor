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


from tensorflow.python.framework import tensor_util
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import dtypes

from ..graph_base import GraphRewriterBase
from ..graph_util import GraphAnalyzer
from ..graph_util import GraphRewriterHelper as Helper


class FuseConvRequantizeTransformer(GraphRewriterBase):
    """Fuse Quantized Conv Op with the successor Requantize Op.
    """
    fuse_patterns = [[
        "QuantizedConv2DWithBiasAndRelu",
        "QuantizedDepthwiseConv2DWithBiasAndRelu",
        "QuantizedConv2DWithBias",
    ], ['RequantizePerChannel', 'Requantize']]
    sum_pattern = [["QuantizedConv2DWithBiasSumAndRelu"], ['RequantizePerChannel', 'Requantize']]

    def __init__(self, model, device='cpu'):
        super().__init__(model)
        self.device = device
        self.graph_analyzer = GraphAnalyzer()
        self.graph_analyzer.graph = self.model

        self.graph_info = self.graph_analyzer.parse_graph()

    def do_transformation(self):
        """Fuse the quantized op with the following requantize op.
            The transformation has two stages, the first step is to fuse the patterns
            defined in self.fuse_patterns and the last step is to fuse the self.sum_patterns.
        Returns:
            [graphdef]: the optimized graphdef object
        """
        int8_type = dtypes.qint8.as_datatype_enum
        uint8_type = dtypes.quint8.as_datatype_enum
        float32_type = dtypes.float32.as_datatype_enum
        qint32_type = dtypes.qint32.as_datatype_enum

        while True:
            target_nodes = self.graph_analyzer.query_fusion_pattern_nodes(self.fuse_patterns)
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
            if 'Tinput' in quantized_node.attr:
                new_node.attr["Tinput"].CopyFrom(quantized_node.attr['Tinput'])
            if 'Tfilter' in quantized_node.attr:
                new_node.attr["Tfilter"].CopyFrom(quantized_node.attr['Tfilter'])
            if 'strides' in quantized_node.attr:
                new_node.attr["strides"].CopyFrom(quantized_node.attr['strides'])
            if 'padding' in quantized_node.attr:
                new_node.attr["padding"].CopyFrom(quantized_node.attr['padding'])

            parent_node_name = Helper.node_name_from_input(quantized_node.input[0])
            max_filter_node = self.graph_info[new_node.input[6]].node
            min_filter_node = self.graph_info[new_node.input[5]].node
            last_node = self.graph_info[new_node.input[0]].node
            if last_node.op.find('Requantize') != -1:
                bias_node = self.graph_info[new_node.input[2]].node
                max_input_node = self.graph_info[last_node.input[-1]].node
                min_input_node = self.graph_info[last_node.input[-2]].node
                min_input = (min_input_node.attr['value'].tensor.float_val)[0]
                max_input = (max_input_node.attr['value'].tensor.float_val)[0]
                if 'Depthwise' in quantized_node_op or requantize_node.op.find('PerChannel') != -1:
                    channel_size = max_filter_node.attr['value'].tensor.tensor_shape.dim[0].size
                    max_filter_tensor = tensor_util.MakeNdarray(
                        min_filter_node.attr['value'].tensor)
                    min_filter_tensor = tensor_util.MakeNdarray(
                        min_filter_node.attr['value'].tensor)
                else:
                    channel_size = 1
                    max_filter_tensor = []
                    min_filter_tensor = []
                    max_filter_tensor.append((max_filter_node.attr['value'].tensor.float_val)[0])
                    min_filter_tensor.append((min_filter_node.attr['value'].tensor.float_val)[0])
                bias_tensor = tensor_util.MakeNdarray(
                    self.graph_info[new_node.input[2]].node.attr['value'].tensor)
                bias_length = bias_tensor.shape[0]
                scales = []
                activation_range = 127.0 if new_node.attr["Tinput"].type == dtypes.qint8 else 255.0
                weights_range = 127.0
                for i in range(channel_size):
                    scales.append(activation_range * weights_range/
                                   (max(abs(max_input), abs(min_input)) *
                                    max(abs(max_filter_tensor[i]), abs(min_filter_tensor[i]))))

                int32_bias = []
                if channel_size > 1:
                    for i in range(bias_length):
                        int32_bias.append((int)(bias_tensor[i] * scales[i]))
                else:
                    for i in range(bias_length):
                        int32_bias.append((int)(bias_tensor[i] * scales[0]))

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

            if "padding_list" in quantized_node.attr:
                new_node.attr["padding_list"].CopyFrom(quantized_node.attr['padding_list'])
            if "dilations" in quantized_node.attr:
                new_node.attr["dilations"].CopyFrom(quantized_node.attr['dilations'])

            if quantized_node.op == "QuantizedConv2D" or \
                    quantized_node.op == "QuantizedConv2DWithBias":
                new_node.attr["out_type"].CopyFrom(attr_value_pb2.AttrValue(type=int8_type))
            else:
                new_node.attr["out_type"].CopyFrom(attr_value_pb2.AttrValue(type=uint8_type))
            self.graph_analyzer.replace_single_node(
                new_node, [parent_node_name], quantized_node_name,
                [self.graph_info[requantize_node_name].outputs[0]], requantize_node_name)
            self.graph_analyzer.remove_node(quantized_node_name)

        target_nodes = self.graph_analyzer.query_fusion_pattern_nodes(self.sum_pattern)
        while target_nodes:
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

            for _, value in enumerate(quantized_node.input[:-1]):
                new_node.input.append(value)

            new_node.attr["Tinput"].CopyFrom(quantized_node.attr['Tinput'])
            new_node.attr["Tfilter"].CopyFrom(quantized_node.attr['Tfilter'])
            new_node.attr["strides"].CopyFrom(quantized_node.attr['strides'])
            new_node.attr["padding"].CopyFrom(quantized_node.attr['padding'])

            new_node.input.append(requested_output_min_name)
            new_node.input.append(requested_output_max_name)
            deq_node = self.graph_info[Helper.node_name_from_input(quantized_node.input[-1])].node
            if deq_node.op != 'Dequantize' or deq_node.op.find("Quantize") != -1:
                self.logger.debug('Dropping fusion due to unsupported pattern..... {}'.format(i))
                target_nodes.remove(i)
                continue
            if deq_node.op == 'Dequantize':
                original_summand_node = self.graph_info[Helper.node_name_from_input(
                    deq_node.input[0])].node
            else:
                original_summand_node = deq_node
            summand_op_type = uint8_type if dtypes.as_dtype(
                deq_node.attr["T"].type) == uint8_type else int8_type

            for j in range(3):
                new_node.input.append(original_summand_node.name + ':{}'.format(j))

            if "padding_list" in quantized_node.attr:
                new_node.attr["padding_list"].CopyFrom(quantized_node.attr['padding_list'])

            if "dilations" in quantized_node.attr:
                new_node.attr["dilations"].CopyFrom(quantized_node.attr['dilations'])
            new_node.attr["out_type"].CopyFrom(attr_value_pb2.AttrValue(type=uint8_type))

            new_node.attr["Tbias"].CopyFrom(attr_value_pb2.AttrValue(type=float32_type))

            if summand_op_type == int8_type:
                new_node.op = "QuantizedConv2DWithBiasSignedSumAndReluAndRequantize"
            new_node.attr["Tsummand"].CopyFrom(attr_value_pb2.AttrValue(type=summand_op_type))

            self.graph_analyzer.replace_single_node(
                new_node, [quantized_node.input[0], original_summand_node.name],
                quantized_node.name, self.graph_info[requantize_node_name].outputs,
                requantize_node_name)
            self.graph_analyzer.remove_node(quantized_node_name)

            if deq_node.op == 'Dequantize':
                self.graph_analyzer.remove_node_with_single_input_output(deq_node.name)
            target_nodes.remove(i)

        return self.graph_analyzer.dump_graph()
