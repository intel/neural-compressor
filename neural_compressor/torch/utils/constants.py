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


# double quant params

DOUBLE_QUANT_CONFIGS = {
    "GGML_TYPE_Q4_K": {
        "weight_dtype": "int",
        "weight_bits": 4,
        "weight_group_size": 32,
        "double_quant_bits": 6,
        "double_quant_dtype": "int",
        "double_quant_sym": True,
        "double_quant_group_size": 8,
    },
    "BNB": {
        "weight_dtype": "nf4",
        "weight_bits": 4,
        "weight_group_size": 32,
        "double_quant_bits": 8,
        "double_quant_dtype": "int",
        "double_quant_sym": False,
        "double_quant_group_size": 256,
    },
}
