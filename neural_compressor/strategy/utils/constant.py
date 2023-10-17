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

from enum import Enum

PRECISION_LIST = ["bf16", "fp16", "fp32"]
QUANT_MODE_SET = {"static", "dynamic"}
LOWER_BIT_LIST = ["int4"]

TUNING_ITEMS_LST = [
    ("activation", "scheme"),
    ("activation", "algorithm"),
    ("activation", "granularity"),
    ("weight", "scheme"),
    ("weight", "algorithm"),
    ("weight", "granularity"),
    "sampling_size",
]
WEIGHT_ONLY_TUNING_ITEMS_LST = [
    ("activation", "scheme"),
    ("activation", "algorithm"),
    ("activation", "granularity"),
    ("weight", "scheme"),
    ("weight", "algorithm"),
    ("weight", "granularity"),
    ("weight", "bits"),
    ("weight", "group_size"),
    ("weight", "dtype"),
    "sampling_size",
]

PRECISION_SET_V2_0 = {"fp32", "bf16", "fp16"}

auto_query_order = ["static", "dynamic", "bf16", "fp16", "fp32"]
static_query_order = ["static", "bf16", "fp16", "fp32"]
dynamic_query_order = ["dynamic", "bf16", "fp16", "fp32"]
auto_query_order_o0 = ["bf16", "fp16", "fp32", "static", "dynamic"]
weight_only_query_order = ["weight_only", "fp32"]


FALLBACK_RECIPES_SET = {
    "first_conv_or_matmul_quantization",
    "last_conv_or_matmul_quantization",
    "pre_post_process_quantization",
}


class WoqTuningParams(Enum):
    """This enumeration class represents the different tuning parameters for the weight only quant (WOQ) algorithm.

    Args:
        Enum (Enum): base enumeration class
    
    Attributes:
        RTN (int): Represents the RTN algorithm, which is a type of WOQ algorithm.
        GPTQ (int): Represents the GPTQ algorithm, which is a type of WOQ algorithm.
        GPTQ_DISABLE_LAST_MATMUL (int): Represents the GPTQ algorithm with the last matrix multiplication disabled.
        GPTQ_GROUP_SIZE_32 (int): Represents the GPTQ algorithm with a group size of 32.
        GPTQ_GROUP_SIZE_128 (int): Represents the GPTQ algorithm with a group size of 128.
        AWQ (int): Represents the AWQ algorithm, which is a type of WOQ algorithm.
    """
    

Attributes:
    RTN (int): Represents the RTN algorithm, which is a type of WOQ algorithm.
    GPTQ (int): Represents the GPTQ algorithm, which is a type of WOQ algorithm.
    GPTQ_DISABLE_LAST_MATMUL (int): Represents the GPTQ algorithm with the last matrix multiplication disabled.
    GPTQ_GROUP_SIZE_32 (int): Represents the GPTQ algorithm with a group size of 32.
    GPTQ_GROUP_SIZE_128 (int): Represents the GPTQ algorithm with a group size of 128.
    AWQ (int): Represents the AWQ algorithm, which is a type of WOQ algorithm.
    RTN = 1
    GPTQ = 2
    GPTQ_DISABLE_LAST_MATMUL = 3
    GPTQ_GROUP_SIZE_32 = 4
    GPTQ_GROUP_SIZE_128 = 5
    AWQ = 6
