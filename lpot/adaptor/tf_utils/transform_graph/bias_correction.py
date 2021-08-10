#
#  -*- coding: utf-8 -*-
#
#  Copyright (c) 2021 Intel Corporation
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


class BiasCorrection(GraphTransformBase):
    """
    This class implements the bias correction graph transform.
    Will correct the weight and scale for *Conv2D* op
    weight_empirical:
    our task is to correct int8 weight distribution close to fp32 weight
    r*(W_int8 + u) -> W_fp32, r is variance ratio between fp32 and int8
    u is the difference between fp32 and int8 channel wise, it's equal to minimize:
    round(scale_c * (W_fp32 + shift))/scale - r*(round(scale * W_fp32) + scale*u)/scale
    notice we can only change the first round: round(scale_c * (W_fp32 + shift))
    an empirical solution is to make:
    scale_c = r * scale and shift = u
    with this we don't change the min/max value, and correct the weight
    """

    def __init__(self, input_graph, fp32_graph, method='weight_empirical'):
        # only support weight_empirical now
        self.bias_correct_map = {'weight_empirical': self._weight_empirical}
        assert method in self.bias_correct_map, \
            'only support weight empirical correction method'

        super(BiasCorrection, self).__init__(input_graph)
        self.fp32_graph = fp32_graph
        self.input_graph = input_graph
        self.method = method
        self.fp32_node_mapping = {}
        self.parse_input_pb()

    def _weight_empirical(self):

        for node in self.fp32_graph.node:
            if node.name not in self.fp32_node_mapping:
                self.fp32_node_mapping[node.name] = node

        for node_name in self.node_mapping:
            node = self.node_mapping[node_name]
            node_op = node.op
            if 'QuantizedConv2D' not in node_op:
                continue

            int8_filter = self.node_mapping[self.get_node_name_from_input(
                node.input[1])]

            int8_value = tensor_util.MakeNdarray(
                int8_filter.attr['value'].tensor)
            tr_int8_value = int8_value.transpose([3, 0, 1, 2])

            fp32_filter_name = self.get_node_name_from_input(
                node.input[1]).split('_qint8_const')[0]
            fp32_filter = self.fp32_node_mapping[fp32_filter_name]

            fp32_value = tensor_util.MakeNdarray(
                fp32_filter.attr['value'].tensor)
            tr_fp32_value = fp32_value.transpose([3, 0, 1, 2])

            # if bias fused, then offset to min/max filter should be 5
            offset = 5 if 'Bias' in node_op else 4
            min_filter_node = self.node_mapping[
                node.input[offset]]
            max_filter_node = self.node_mapping[
                node.input[offset + 1]]

            channel_size = 1 if not min_filter_node.attr[
                'value'].tensor.tensor_shape.dim else min_filter_node.attr[
                    'value'].tensor.tensor_shape.dim[0].size

            if channel_size == 1:
                max_filter_tensor = []
                min_filter_tensor = []
                max_filter_tensor.append(
                    (max_filter_node.attr['value'].tensor.float_val)[0])
                min_filter_tensor.append(
                    (min_filter_node.attr['value'].tensor.float_val)[0])
            else:
                max_filter_tensor = tensor_util.MakeNdarray(
                    max_filter_node.attr['value'].tensor)
                min_filter_tensor = tensor_util.MakeNdarray(
                    min_filter_node.attr['value'].tensor)

            tr_quantized_fp32_value = np.zeros_like(tr_fp32_value)
            tr_corrected_int8_value = np.zeros_like(tr_int8_value)

            for i in range(channel_size):
                scale = max(abs(max_filter_tensor[i]),
                            abs(min_filter_tensor[i])) / 127
                tr_quantized_fp32_value[i] = tr_int8_value[i].astype(np.float64) * scale
                delta_mean = np.mean((tr_fp32_value[i] - tr_quantized_fp32_value[i]).flatten())
                var_ratio = np.std(tr_fp32_value[i].flatten()) / \
                    np.std(tr_quantized_fp32_value[i].flatten()) if \
                    np.std(tr_quantized_fp32_value[i].flatten()) != 0 else 1
                tr_corrected_int8_value[i] = (var_ratio / scale) * (tr_fp32_value[i] + delta_mean)

            correct_int8_value = tr_int8_value.transpose([1, 2, 3, 0])
            assert int8_value.shape == correct_int8_value.shape, \
                'correct filter shape should equal with origin filter shape'
            bias = int8_value.astype(np.float32) - correct_int8_value.astype(np.float32)
            if np.sum(bias) != 0 :
                int8_filter.attr['value'].CopyFrom(
                    attr_value_pb2.AttrValue(
                        tensor=tensor_util.make_tensor_proto(
                            correct_int8_value, dtypes.qint8, int8_value.shape)))
        return self.input_graph

    def do_transformation(self):
        return self.bias_correct_map[self.method]()


