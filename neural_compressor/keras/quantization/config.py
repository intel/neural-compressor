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

from __future__ import annotations

from enum import Enum
from typing import Callable, Dict, List, NamedTuple, Union

import tensorflow as tf

from neural_compressor.common.base_config import BaseConfig, register_config, registered_configs
from neural_compressor.common.utility import KERAS_STATIC_QUANT
from neural_compressor.config import PostTrainingQuantConfig

FRAMEWORK_NAME = "keras"


class Backend(Enum):
    DEFAULT = "keras"
    ITEX = "itex"


@register_config(framework_name=FRAMEWORK_NAME, algo_name=KERAS_STATIC_QUANT)
class KerasStaticQuantConfig(PostTrainingQuantConfig):
    """Config class for keras static quantization."""
    name = KERAS_STATIC_QUANT


def get_default_keras_config() -> KerasStaticQuantConfig:
    """Generate the default keras config.

    Returns:
        the default keras config.
    """
    return KerasStaticQuantConfig()
