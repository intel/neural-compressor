# -*- coding: utf-8 -*-
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
"""Intel Neural Compressor Profiling."""

# double quant params

DOUBLE_QUANT_CONFIGS = {
    "DEFAULT" : {
        "double_quant_bits": 8,
        "double_quant_dtype": "fp32",
        "double_quant_sym": True,
        "double_quant_group_size": 256,
    },
    "GGML_TYPE_Q4_K" : {
        "double_quant_bits": 6,
        "double_quant_dtype": "int",
        "double_quant_sym": True,
        "double_quant_group_size": 8,
    },
    "BNB" : {
        "double_quant_bits": 8,
        "double_quant_dtype": "int",
        "double_quant_sym": True,
        "double_quant_group_size": 8,
    }
}