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
from copy import deepcopy

import numpy as np
from .algorithm import Algorithm, algorithm_registry

@algorithm_registry(algorithm_type='weight_correction')
class WeightCorrection(Algorithm):
    """
    correct int8 weight distribution close to fp32 weight
    r*(W_int8 + u) -> W_fp32, r is variance ratio between fp32 and int8
    u is the difference between fp32 and int8 channel wise, it's equal to minimize:
    round(scale_c * (W_fp32 + shift))/scale - r*(round(scale * W_fp32) + scale*u)/scale
    notice we can only change the first round: round(scale_c * (W_fp32 + shift))
    an empirical solution is to make:
    scale_c = r * scale and shift = u
    with this we don't change the min/max value, and correct the weight
    """
    def __init__(self, eps=1e-5, channel_axis=1):
        self.eps = eps
        self.channel_axis = channel_axis

    def __call__(self, origin_model, q_model, adaptor, dataloader, iterations):

        # (TODO) assume int8 model also use fp32 op list
        # in adaptor fp32 op will be mapped to corresponding int8 op
        graph_info = origin_model.graph_info
        op_list = [op_name for op_name, op_type in graph_info.items() if 'conv' in op_type.lower()]

        #(TODO) assume the weight format should be(oc, ic, h, w)
        fp32_data = adaptor.inspect_tensor(origin_model, dataloader, op_list=op_list, \
            iteration_list=list(range(1, iterations+1)), inspect_type='weight')
        q_data = adaptor.inspect_tensor(q_model, dataloader, op_list=op_list, \
            iteration_list=list(range(1, iterations+1)), inspect_type='weight')

        fp32_weights = fp32_data['weight']
        q_weights = q_data['weight']

        tensor_dict = {}
        # for fp32_op, q_op in node_mapping.items():
        for fp32_op in op_list:
            # (TODO) assume adaptor will map the fp32_op to q_op, so directly assign here
            q_op = fp32_op
            #(TODO) assume fp32 op output and weight all mapped from the first node name
            # fp32 op and quantized op should all have bias
            if fp32_op not in fp32_weights or not len(fp32_weights[fp32_op]) >= 1: 
                continue

            fp32_weight, fp32_weight_name = None, ''
            fp32_bias, fp32_bias_name = None, ''
            for name, value in fp32_weights[fp32_op].items():
                if len(value.shape) > 1:
                    fp32_weight = value
                    fp32_weight_name = name
                if len(value.shape) == 1:
                    fp32_bias = value
                    fp32_bias_name = name

            q_weight, q_weight_name = None, ''
            q_bias, q_bias_name = None, ''
            for name, value in q_weights[q_op].items():
                if len(value.shape) > 1:
                    q_weight = value
                    q_weight_name = name
                if len(value.shape) == 1:
                    q_bias = value
                    q_bias_name = name

            # (fp32_node_name, fp32_weight), fp32_bias = fp32_weights[fp32_op].items()
            # (q_node_name, q_weight), q_bias = q_weights[q_op].items()

            channel_shape = list(range(len(fp32_weight.shape)))
            transpose_shape = [channel_shape.pop(self.channel_axis)]
            transpose_shape.extend(channel_shape)
            t_fp32_weight = np.transpose(fp32_weight, transpose_shape)
            t_fp32_weight = t_fp32_weight.reshape(t_fp32_weight.shape[0], -1)
            t_q_weight = np.transpose(q_weight, transpose_shape)
            t_q_weight = t_q_weight.reshape(t_q_weight.shape[0], -1)

            channel_variance = np.std(t_fp32_weight, axis=1) / \
              (np.std(t_q_weight, axis=1) + self.eps)          

            broad_shape = np.ones(len(fp32_weight.shape), dtype=np.int)
            broad_shape[self.channel_axis] = len(channel_variance)
            channel_variance = channel_variance.reshape(broad_shape)
            variance_q_weight = q_weight * channel_variance
            variance_q_weight = np.transpose(variance_q_weight, transpose_shape)
            variance_q_weight = variance_q_weight.reshape(\
                variance_q_weight.shape[0], -1)

            channel_mean = np.mean(t_fp32_weight, axis=self.channel_axis) - \
                np.mean(variance_q_weight, axis=self.channel_axis)

            channel_mean = channel_mean.reshape(broad_shape)
            tensor_dict[q_weight_name] = channel_variance * fp32_weight + channel_mean

        if len(tensor_dict) > 0:
            adaptor.set_tensor(q_model, tensor_dict)
        return q_model
