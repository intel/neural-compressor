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
from ..utils import logger

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
    def __init__(self, safety_eps=1e-5):
        self.safety_eps = safety_eps

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

            axis = tuple(range(1, len(fp32_weight.shape)))
            variance_per_channel = np.std(fp32_weight, axis=axis) / (
                np.std(q_weight, axis=axis) + self.safety_eps)
            broadcast_axes = (...,) + (np.newaxis,) * (len(fp32_weight.shape) - 1)
            variance_per_channel = variance_per_channel[broadcast_axes]
            corrected_weight = q_weight * variance_per_channel
            mean_per_channel = np.mean(fp32_weight, axis=axis) - np.mean( \
                corrected_weight, axis=axis)
            mean_per_channel = mean_per_channel[broadcast_axes]
            tensor_dict[q_weight_name] = fp32_weight * variance_per_channel + mean_per_channel

        if len(tensor_dict) > 0:
            adaptor.set_tensor(q_model, tensor_dict)
        return q_model
