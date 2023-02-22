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

"""Constants used for the configuration."""

FP32 = {'weight': {'dtype': ['fp32']}, 'activation': {'dtype': ['fp32']}}
BF16 = {'weight': {'dtype': ['bf16']}, 'activation': {'dtype': ['bf16']}}

INT8_SYM_MINMAX_PERCHANNEL = {'dtype': ['int8'],
                              'scheme': ['sym'],
                              'algorithm': ['minmax'],
                              'granularity': ['per_channel']}

INT8_SYM_MINMAX_PERTENSOR = {'dtype': ['int8'],
                             'scheme': ['sym'],
                             'algorithm': ['minmax'],
                             'granularity': ['per_tensor']}
                             
INT8_SYM_KL_PERTENSOR = {'dtype': ['int8'],
                         'scheme': ['sym'],
                         'algorithm': ['kl'],
                         'granularity': ['per_tensor']}
                              
INT8_SYM_KL_PERCHANNEL = {'dtype': ['int8'],
                          'scheme': ['sym'],
                          'algorithm': ['kl'],
                          'granularity': ['per_channel']}

UINT8_ASYM_MINMAX_PERCHANNEL = {'dtype': ['uint8'],
                                'scheme': ['asym'],
                                'algorithm': ['minmax'],
                                'granularity': ['per_channel']}

UINT8_ASYM_MINMAX_PERTENSOR = {'dtype': ['uint8'],
                               'scheme': ['asym'],
                               'algorithm': ['minmax'],
                               'granularity': ['per_tensor']}
                             
UINT8_ASYM_KL_PERTENSOR = {'dtype': ['uint8'],
                           'scheme': ['asym'],
                           'algorithm': ['kl'],
                           'granularity': ['per_tensor']}
                              
UINT8_ASYM_KL_PERCHANNEL = {'dtype': ['uint8'],
                            'scheme': ['asym'],
                            'algorithm': ['kl'],
                            'granularity': ['per_channel']}


# Options for recipes, the first options is the default value.
RECIPES = {
    "common":{
        'smooth_quant': [False, True],
        'fast_bias_correction' : [False, True],
        'weight_correction' : [False, True],},
    "tensorflow": {
        'first_conv_or_matmul_quantization' : [True, False],
        'last_conv_or_matmul_quantization' : [True, False],
        'scale_propagation_max_pooling' : [True, False],
        'scale_propagation_concat' : [True, False],
        },
    "onnxruntime": {
        'first_conv_or_matmul_quantization' : [True, False],
        'last_conv_or_matmul_quantization' : [True, False],
        'pre_post_process_quantization' : [True, False],
        'gemm_to_matmul' : [False, True],
        'graph_optimization_level' : ['DISABLE_ALL', 'ENABLE_BASIC', 'ENABLE_EXTENDED', 'ENABLE_ALL'],
        'add_qdq_pair_to_weight' : [False, True],
        'optypes_to_exclude_output_quant' : [[]], # TODO
        'dedicated_qdq_pair' : [True, False]},
    "pytorch": {},
}

RECIPES_PRIORITY = [
    "smooth_quant",
    "scale_propagation_max_pooling",
    "scale_propagation_concat",
    "first_conv_or_matmul_quantization",
    "last_conv_or_matmul_quantization",
    "pre_post_process_quantization",
    "fast_bias_correction",
    "weight_correction",
    "gemm_to_matmul",
    "graph_optimization_level",
    "add_qdq_pair_to_weight",
    "optypes_to_exclude_output_quan",
    "dedicated_qdq_pair"]