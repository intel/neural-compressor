"""Pruners."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
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

from os.path import dirname, basename, isfile, join
import glob

from .base import PATTERNS

modules = glob.glob(join(dirname(__file__), "*.py"))

for f in modules:
    if isfile(f) and not f.startswith("__") and not f.endswith("__init__.py"):
        __import__(basename(f)[:-3], globals(), locals(), level=1)

FRAMEWORK = {"pytorch": "pt", "keras": "keras"}


def get_pattern(config, modules, framework="pytorch"):
    """Get registered pattern class.

    Get a Pattern object from PATTERNS.

    Args:
        config: A config dict object that contains the pattern information.
        modules: Torch neural network modules to be pruned with the pattern.

    Returns:
        A Pattern object.

    Raises:
        AssertionError: Currently only support patterns which have been registered in PATTERNS.
    """
    assert framework in FRAMEWORK.keys(), f"does not support {framework}, currently only support {FRAMEWORK.keys()}"

    name = config.pattern
    name = name.split("_")[-1]
    pattern = FRAMEWORK[framework]
    if "x" in name:
        pattern += "NxM"
    elif ":" in name:
        pattern += "N:M"
    elif "mha" in name:
        pattern += "MHA"
    else:
        assert False, f"currently only support {PATTERNS.keys()}"
    if pattern not in PATTERNS.keys():
        assert False, f"currently only support {PATTERNS.keys()}"
    return PATTERNS[pattern](config, modules)
