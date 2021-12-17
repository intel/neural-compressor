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

