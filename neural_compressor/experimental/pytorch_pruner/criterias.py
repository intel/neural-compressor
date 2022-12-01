"""pruning criteria."""
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
import torch

CRITERIAS = {}


def register_criteria(name):
    """Register a criteria to the registry."""

    def register(criteria):
        CRITERIAS[name] = criteria
        return criteria

    return register


def get_criteria(config, modules):
    """Get registered criteria class."""
    name = config["criteria_type"]
    if name not in CRITERIAS.keys():
        assert False, f"criterias does not support {name}, currently only support {CRITERIAS.keys()}"
    return CRITERIAS[name](modules, config)


class Criteria:
    """Pruning criteria.

    Args:
        config: A config dict object that includes information about pruner and pruning criteria.
        modules: A dict {"module_name": Tensor} that stores the pruning modules' weights.
    
    Attributes:
        scores: A dict {"module_name": Tensor} that stores the scores of pruning modules.
    """

    def __init__(self, modules, config):
        """Initiliaze a pruning criterion."""
        self.scores = {}
        self.modules = modules
        self.config = config

    def on_step_begin(self):
        """Calculate and store the pruning scores of pruning modules at the beginning of a step."""
        pass

    def on_after_optimizer_step(self):
        """Calculate and store the pruning scores of pruning modules after the optimizer step."""
        pass


@register_criteria('magnitude')
class MagnitudeCriteria(Criteria):
    """Pruning criterion.
    
    The magnitude criteria_class is derived from Criteria. 
    The magnitude value is used to score and determine if a weight is to be pruned.

    Args:
        config: A config dict object that includes information about pruner and pruning criteria.
        modules: A dict {"module_name": Tensor} that stores the pruning modules' weights.
    
    Attributes:
        scores: A dict {"module_name": Tensor} that stores the scores of pruning modules.
    """

    def __init__(self, modules, config):
        """Initiliaze a magnitude pruning criterion."""
        super(MagnitudeCriteria, self).__init__(modules, config)

    def on_step_begin(self):
        """Calculate and store the pruning scores based on magtinude criterion."""
        with torch.no_grad():
            for key in self.modules.keys():
                p = self.modules[key].weight.data
                self.scores[key] = p

@register_criteria('gradient')
class GradientCriteria(Criteria):
    """Pruning criterion.
    
    The gradient criteria_class is derived from Criteria. 
    The absolute value of gradient is used to score and determine if a weight is to be pruned.

    Args:
        config: A config dict object that includes information about pruner and pruning criteria.
        modules: A dict {"module_name": Tensor} that stores the pruning modules' weights.
    
    Attributes:
        scores: A dict {"module_name": Tensor} that stores the scores of pruning modules.
    """

    def __init__(self, modules, config):
        """Initiliaze a gradient pruning criterion."""
        super(GradientCriteria, self).__init__(modules, config)

    def on_after_optimizer_step(self):
        """Calculate and store the pruning scores based on gradient criterion."""
        with torch.no_grad():
            for key in self.modules.keys():
                p = self.modules[key].weight
                self.scores[key] = torch.abs(p.grad)

@register_criteria('snip')
class SnipCriteria(Criteria):
    """Pruning criterion.
    
    The snip criteria_class is derived from Criteria. 
    The product of magnitude and gradient is used to score and determine if a weight is to be pruned.
    Please refer to SNIP: Single-shot Network Pruning based on Connection Sensitivity.
    (https://arxiv.org/abs/1810.02340)

    Args:
        config: A config dict object that includes information about pruner and pruning criteria.
        modules: A dict {"module_name": Tensor} that stores the pruning modules' weights.
    
    Attributes:
        scores: A dict {"module_name": Tensor} that stores the scores of pruning modules.
    """

    def __init__(self, modules, config):
        """Initiliaze a snip pruning criterion."""
        super(SnipCriteria, self).__init__(modules, config)
        assert self.config.end_step > 0, "gradient based criteria does not work on step 0"

    def on_after_optimizer_step(self):
        """Calculate and store the pruning scores based on snip criterion."""
        ##self.mask_weights()
        with torch.no_grad():
            for key in self.modules.keys():
                p = self.modules[key].weight
                self.scores[key] = torch.abs(p * p.grad)


@register_criteria('snip_momentum')
class SnipMomentumCriteria(Criteria):
    """Pruning criterion.
    
    The snip_momentum criteria_class is derived from Criteria. 
    A momentum mechanism is used to calculate snip score, which determines if a weight is to be pruned.

    Args:
        config: A config dict object that includes information about pruner and pruning criteria.
        modules: A dict {"module_name": Tensor} that stores the pruning modules' weights.
        alpha: A parameter that determines how much of the snip score is preserved from last pruning step.
        beta: A parameter that determines how much of the snip score is updated at the current step.
    
    Attributes:
        scores: A dict {"module_name": Tensor} that stores the scores of pruning modules.
    """

    def __init__(self, modules, config):
        """Initiliaze a snip_momentum pruning criterion."""
        super(SnipMomentumCriteria, self).__init__(modules, config)
        assert self.config.end_step > 0, "gradient based criteria does not work on step 0"
        for key in modules.keys():
            p = modules[key].weight
            self.scores[key] = torch.zeros(p.shape).to(p.device)

        self.alpha = 0.9
        self.beta = 1.0

    def on_after_optimizer_step(self):
        """Calculate and store the pruning scores based on snip_momentum criterion."""
        with torch.no_grad():
            for key in self.modules.keys():
                p = self.modules[key].weight
                self.scores[key] *= self.alpha
                self.scores[key] += self.beta * torch.abs(p * p.grad)
