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
"""Utils: provide useful methods and auxiliary functionalities."""

from .collect_layer_histogram import LayerHistogramCollector
from .logger import log, info, debug, warn, warning, error, fatal
from .options import OPTIONS
from .utility import alias_param

__all__ = [
    "LayerHistogramCollector",
    "log",
    "info",
    "debug",
    "warn",
    "warning",
    "error",
    "fatal",
    "OPTIONS",
    "alias_param",
]
