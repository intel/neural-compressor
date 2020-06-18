#
#  -*- coding: utf-8 -*-
#
#  Copyright (c) 2019 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.python.framework import tensor_util
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import dtypes
from .graph_transform_base import GraphTransformBase


class RerangeQuantizedConcat(GraphTransformBase):
    """
    This class implements the rerange_quantize concat graph transform.
    """
    fused_requantized_bias_op = ("QuantizedConv2DWithBiasAndRequantize",
                                 "QuantizedConv2DWithBiasAndReluAndRequantize",
                                 "QuantizedConv2DWithBiasSumAndReluAndRequantize",
                                 "QuantizedConv2DWithBiasSignedSumAndReluAndRequantize")

    offset_map = {"QuantizedConv2DAndRequantize": 6,
                  "QuantizedConv2DAndReluAndRequantize": 6,
                  "QuantizedConv2DWithBiasAndRequantize": 7,
                  "QuantizedConv2DWithBiasAndReluAndRequantize": 7,
                  "QuantizedConv2DWithBiasSumAndReluAndRequantize": 7,
                  "QuantizedConv2DWithBiasSignedSumAndReluAndRequantize": 7}

    def __init__(self, input_pb):
        super(RerangeQuantizedConcat, self).__init__(input_pb)

        self.parse_input_pb()
        self.concat_node_input_mapping = {}

    def _analyze_concat_node(self):
        """
        Parse the graph to get concat's node inputs.
        Returns:
            The dict that key is concat node's name while value are the
            concat op's input nodeDef.
        """

        for _, node in enumerate(self.input_graph.node):
            if node.op == "QuantizedConcatV2":
                concat_node_input_node = []
                for i in range(node.attr['N'].i):
                    input_node = self.node_mapping[self.get_node_name_from_input(node.input[i])]
                    input_node_op = input_node.op
                    if input_node_op in self.offset_map:
                        concat_node_input_node.append(input_node)

                self.concat_node_input_mapping[node.name] = concat_node_input_node

    def _calc_concat_scale(self):
        """
        Update the all conv's op scale which connected to concat op.
        """
        for node_name in self.concat_node_input_mapping:
            combined_min = np.finfo(np.float64).max
            combined_max = -combined_min
            for node_input in self.concat_node_input_mapping[node_name]:
                offset_value = self.offset_map[node_input.op]
                min_value_node = self.node_mapping[node_input.input[offset_value]]
                max_value_node = self.node_mapping[node_input.input[offset_value + 1]]
                min_value = min_value_node.attr['value'].tensor.float_val[0]
                max_value = max_value_node.attr['value'].tensor.float_val[0]
                if min_value < combined_min:
                    combined_min = min_value
                if max_value > combined_max:
                    combined_max = max_value

            for node_input in self.concat_node_input_mapping[node_name]:
                offset_value = self.offset_map[node_input.op]
                min_value_node = self.node_mapping[node_input.input[offset_value]]
                max_value_node = self.node_mapping[node_input.input[offset_value + 1]]
                min_value_node.attr["value"].CopyFrom(
                    attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
                        float(combined_min), dtypes.float32, [])))

                max_value_node.attr["value"].CopyFrom(
                    attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
                        float(combined_max), dtypes.float32, [])))

    def _update_bias(self):
        """
        Convert the bias from float to int.
        """
        for node_name in self.node_mapping:
            current_node = self.node_mapping[node_name]
            current_node_op = current_node.op
            if current_node_op in self.fused_requantized_bias_op:
                bias_node = self.node_mapping[self.get_node_name_from_input(current_node.input[2])]
                bias_node_type = current_node.attr['Tbias']

                if bias_node_type.type != dtypes.float32 or bias_node_type.type == dtypes.qint32:
                    continue
                input_node_name = self.get_node_name_from_input(current_node.input[0])
                if self.node_mapping[input_node_name].op == "QuantizeV2":
                    continue

                found_last_conv_flag = False
                input_node = current_node
                last_conv_node = None

                while not found_last_conv_flag:
                    input_node = self.node_mapping[self.get_node_name_from_input(input_node.input[0])]
                    if input_node.op in self.offset_map:
                        found_last_conv_flag = True
                        last_conv_node = input_node
                    elif input_node.op in "QuantizedConcatV2":
                        found_last_conv_flag = False
                    elif input_node.op not in ("QuantizedMaxPool", "QuantizedAvgPool",):
                        found_last_conv_flag = True

                if not last_conv_node:
                    continue

                min_filter_node = self.node_mapping[current_node.input[5]]
                max_filter_node = self.node_mapping[current_node.input[6]]
                min_filter = min_filter_node.attr['value'].tensor.float_val[0]
                max_filter = max_filter_node.attr['value'].tensor.float_val[0]
                offset_value = self.offset_map[current_node_op]
                min_freezed_output_node = self.node_mapping[last_conv_node.input[offset_value]]
                max_freezed_output_node = self.node_mapping[last_conv_node.input[offset_value + 1]]
                min_input = min_freezed_output_node.attr['value'].tensor.float_val[0]
                max_input = max_freezed_output_node.attr['value'].tensor.float_val[0]

                bias_scale = 255.0 * 127.0 / (
                    max(abs(max_input), abs(min_input)) * max(abs(max_filter), abs(min_filter)))

                bias_tensor = (tensor_util.MakeNdarray(bias_node.attr['value'].tensor))
                bias_length = bias_tensor.shape[0]
                q_bias = []
                for i in range(bias_length):
                    q_bias.append(int(bias_tensor[i] * bias_scale))
                current_node.attr['Tbias'].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.qint32.as_datatype_enum))
                bias_node.attr['dtype'].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.qint32.as_datatype_enum))

                bias_node.attr['value'].CopyFrom(attr_value_pb2.AttrValue(
                    tensor=tensor_util.make_tensor_proto(q_bias, dtypes.int32, bias_tensor.shape)))
                bias_node.attr['value'].tensor.dtype = dtypes.qint32.as_datatype_enum

    def do_transformation(self):
        """
        Execute the rerange_quantized_concat graph transformation.
        Return: Transformed grafhdef
        """
        self._analyze_concat_node()
        if self.concat_node_input_mapping:  # Concat node is found and need to do transformation.
            self._calc_concat_scale()
            self._update_bias()
        return self.input_graph
