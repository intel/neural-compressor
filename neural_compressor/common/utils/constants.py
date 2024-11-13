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
"""All frameworks-agnostic constants."""

# constants for configs
GLOBAL = "global"
LOCAL = "local"
DEFAULT_WHITE_LIST = "*"
EMPTY_WHITE_LIST = None

# config name
BASE_CONFIG = "base_config"
COMPOSABLE_CONFIG = "composable_config"
RTN = "rtn"
STATIC_QUANT = "static_quant"
SMOOTH_QUANT = "smooth_quant"
GPTQ = "gptq"
AWQ = "awq"  # pragma: no cover
HQQ = "hqq"  # pragma: no cover
TEQ = "teq"  # pragma: no cover
AUTOROUND = "autoround"
FP8_QUANT = "fp8_quant"
MX_QUANT = "mx_quant"
MIXED_PRECISION = "mixed_precision"

# options
import datetime

DEFAULT_WORKSPACE = "./nc_workspace/{}/".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

from typing import Callable, Union

OP_NAME_OR_MODULE_TYPE = Union[str, Callable]

# mode name
from enum import Enum


class Mode(Enum):
    """Enumeration class representing different modes of the quantizer execution."""

    PREPARE = "prepare"
    CONVERT = "convert"
    QUANTIZE = "quantize"
    LOAD = "load"


SERVER_PROCESSOR_BRAND_KEY_WORLD_LST = ["Xeon"]
