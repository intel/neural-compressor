# -*- coding: utf-8 -*-
# Copyright (c) 2024 Intel Corporation
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
"""Intel Neural Compressor PyTorch constants."""

# double quant params
DOUBLE_QUANT_CONFIGS = {
    "BNB_NF4": {
        "dtype": "nf4",
        "bits": 4,
        "group_size": 32,
        "use_double_quant": True,
        "double_quant_bits": 8,
        "double_quant_dtype": "int",
        "double_quant_use_sym": False,
        "double_quant_group_size": 256,
    },
    # TODO: (Xin) current implementation is not the same as GGML.
    # GGML is using double_quant_bits to quantize zero points
    "GGML_TYPE_Q4_K": {
        "dtype": "int",
        "bits": 4,
        "use_sym": False,
        "group_size": 32,
        "use_double_quant": True,
        "double_quant_bits": 6,
        "double_quant_dtype": "int",
        "double_quant_use_sym": True,
        "double_quant_group_size": 8,
    },
}

# Setting priorities for algorithms, a higher number indicates a higher priority.
PRIORITY_GPTQ = 90
PRIORITY_RTN = 80
PRIORITY_HQQ = 75
PRIORITY_AWQ = 70
PRIORITY_TEQ = 60
PRIORITY_AUTOROUND = 50


PT2E_STATIC_QUANT = "pt2e_static_quant"
PT2E_DYNAMIC_QUANT = "pt2e_dynamic_quant"


# load format name
from enum import Enum


class SaveLoadFormat(Enum):
    """Load API formats for different usages."""

    DEFAULT = "default"
    HUGGINGFACE = "huggingface"


LM_HEAD_NAMES = [".*lm_head", ".*output_layer", ".*embed_out"]
