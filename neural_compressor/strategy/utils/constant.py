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


WOQ_TUNING_ALGOS = {
    "RTN_G32ASYM": {"algorithm": "RTN", "group_size": 32, "scheme": "asym"},
    "GPTQ_G32ASYM": {"algorithm": "GPTQ", "group_size": 32, "scheme": "asym"},
    "GPTQ_G32ASYM_DISABLE_LAST_MATMUL": {"algorithm": "GPTQ", "group_size": 32, "scheme": "asym"},
    "GPTQ_G128ASYM": {"algorithm": "GPTQ", "group_size": 128, "scheme": "asym"},
    "AWQ_G32ASYM": {"algorithm": "AWQ", "group_size": 32, "scheme": "asym"},
}
