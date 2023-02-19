#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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

"""Strategy constant."""

PRECISION_SET = {'bf16', 'fp32'}
QUANT_MODE_SET = {'static', 'dynamic'}
QUNAT_BIT_SET = {'int8', 'uint8', 'int4', 'uint4'}

TUNING_ITEMS_LST = [('activation','scheme'), ('activation','algorithm'), ('activation','granularity'),
                    ('weight','scheme'), ('weight','algorithm'), ('weight','granularity'),
                    'sampling_size']

TUNING_ITEM_SET = {'scheme', 'algorithm', 'granularity'}

PRECISION_SET_V2_0 = {'fp32', 'bf16'}

auto_query_order = [('static', 'int8'), ('dynamic', 'int8'), ('precision', 'bf16'), \
                    ('precision', 'fp16'), ('precision', 'fp32')]
static_query_order = [('static', 'int8'), ('precision', 'bf16'), ('precision', 'fp16'), ('precision', 'fp32')]
dynamic_query_order = [('dynamic', 'int8'), ('precision', 'bf16'), ('precision', 'fp16'), ('precision', 'fp32')]


from enum import IntEnum

class PostTrainingQuantType(IntEnum):
    """Post training quantization type."""
    STATIC = 0
    DYNAMIC = 1
    WEIGHT_ONLY = 2

"""
Query path format.
(q/p_type, ((a_bits, a_signed), (w_bits,  w_signed )))
('static', (('int8', 'signed'), ('int4', 'unsigned')))
('static', (('int8', 'signed'),                     ))
('static', ( 'int8'                                  )
('static', 'int8')
('static')


"""