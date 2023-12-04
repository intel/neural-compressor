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

PRECISION_SET = {
    "bf16",
    "fp16",
    "fp32",
}
QUANT_MODE_SET = {"static", "dynamic"}
QUNAT_BIT_SET = {"int8", "uint8", "int4", "uint4"}

TUNING_ITEMS_LST = [
    ("activation", "scheme"),
    ("activation", "algorithm"),
    ("activation", "granularity"),
    ("weight", "scheme"),
    ("weight", "algorithm"),
    ("weight", "granularity"),
    "sampling_size",
]

PRECISION_SET_V2_0 = {"fp32", "bf16"}

auto_query_order = ["static", "dynamic", "bf16", "fp16", "fp32"]
static_query_order = ["static", "bf16", "fp16", "fp32"]
dynamic_query_order = ["dynamic", "bf16", "fp16", "fp32"]


FALLBACK_RECIPES_SET = {
    "first_conv_or_matmul_quantization",
    "last_conv_or_matmul_quantization" "pre_post_process_quantization",
}
