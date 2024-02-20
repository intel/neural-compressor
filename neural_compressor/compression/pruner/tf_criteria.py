"""Tensorflow pruning criterion."""

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

import numpy as np

CRITERIA = {}


def register_criterion(name):
    """Register a criterion to the registry."""

    def register(criterion):
        CRITERIA[name] = criterion
        return criterion

    return register


def get_tf_criterion(config, modules):
    """Get registered criterion class."""
    name = config["criterion_type"]
    if name not in CRITERIA.keys():
        assert False, f"criteria does not support {name}, currently only support {CRITERIA.keys()}"
    return CRITERIA[name](modules, config)


class PruningCriterion:
    """Pruning base criterion.

    Args:
        config: A config dict object that includes information about pruner and pruning criterion.
        modules: A dict {"module_name": Tensor} that stores the pruning modules' weights.

    Attributes:
        scores: A dict {"module_name": Tensor} that stores the scores of pruning modules.
    """

    def __init__(self, modules, config):
        """Initialize a pruning criterion."""
        self.scores = {}
        self.modules = modules
        self.config = config

    def on_step_begin(self):
        """Calculate and store the pruning scores of pruning modules at the beginning of a step."""
        pass

    def on_before_optimizer_step(self):
        """Calculate and store the pruning scores of pruning modules before the optimizer step."""
        pass

    def on_after_optimizer_step(self):
        """Calculate and store the pruning scores of pruning modules after the optimizer step."""
        pass


@register_criterion("magnitude")
class MagnitudeCriterion(PruningCriterion):
    """Pruning criterion.

    The magnitude criterion_class is derived from PruningCriterion.
    The magnitude value is used to score and determine if a weight is to be pruned.

    Args:
        config: A config dict object that includes information about pruner and pruning criterion.
        modules: A dict {"module_name": Tensor} that stores the pruning modules' weights.

    Attributes:
        scores: A dict {"module_name": Tensor} that stores the scores of pruning modules.
    """

    def __init__(self, modules, config):
        """Initialize a magnitude pruning criterion."""
        super(MagnitudeCriterion, self).__init__(modules, config)

    def on_step_begin(self):
        """Calculate and store the pruning scores based on a magnitude criterion."""

        for key in self.modules.keys():
            p = self.modules[key].get_weights()[0]
            self.scores[key] = np.abs(p)
