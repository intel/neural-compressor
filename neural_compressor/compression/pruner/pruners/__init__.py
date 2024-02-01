"""Pruning patterns."""

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
from .base import PRUNERS
from ..criteria import CRITERIA

modules = glob.glob(join(dirname(__file__), "*.py"))

for f in modules:
    if isfile(f) and not f.startswith("__") and not f.endswith("__init__.py"):
        __import__(basename(f)[:-3], globals(), locals(), level=1)

FRAMEWORK = {"pytorch": "pt", "keras": "keras"}


def parse_valid_pruner_types():
    """Get all valid pruner names."""
    valid_pruner_types = []
    for x in CRITERIA.keys():
        for p in ["", "_progressive"]:
            valid_pruner_types.append(x + p)
    valid_pruner_types.append("pattern_lock")
    return valid_pruner_types


def get_pruner(config, modules, framework="pytorch"):
    """Get registered pruner class.

    Get a Pruner object from PRUNERS.

    Args:
        modules: A dict {"module_name": Tensor} that stores the pruning modules' weights.
        config: A config dict object that contains the pruner information.

    Returns:
        A Pruner object.

    Raises: AssertionError: Currently only support pruners that have been registered in PRUNERS.
    """
    # do the ugly work here
    # check if it is doing self-multihead-attention pruning
    assert (
        framework in FRAMEWORK.keys()
    ), f"does not support {framework}, currently only support framework: {FRAMEWORK.keys()}"

    if "mha" in config["pattern"]:
        assert framework == "pytorch", "Currently mha only support pytorch framework."
        return PRUNERS[f"{FRAMEWORK[framework]}_mha"](config, modules)
    # if enable progressive pruning or not.
    if "progressive" not in config["pruning_type"]:
        name = config["pruning_type"]
        config["progressive"] = False
    else:
        # if progressive, delete "progressive" words and reset config["progressive"]
        name = config["pruning_type"][0:-12]
        config["progressive"] = True
    if name in CRITERIA:
        if config["progressive"] is False:
            config["criterion_type"] = name
            if "block" in name or "free" in name:
                assert ":" not in config["pattern"], f"{name} pruner type does not support {config['pattern']} pattern."
            else:
                name = "basic"  # return the basic pruner
        else:
            config["criterion_type"] = name
            # name = "progressive"
            # return the progressive pruner
            name = "progressive"

    name = f"{FRAMEWORK[framework]}_{name}"
    if name not in PRUNERS.keys():
        assert False, f"does not support {name}, currently only support {parse_valid_pruner_types()}"
    return PRUNERS[name](config, modules)
