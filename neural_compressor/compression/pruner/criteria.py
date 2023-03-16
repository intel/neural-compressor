"""pruning criterion."""
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
from .utils import torch


CRITERIA = {}


def register_criterion(name):
    """Register a criterion to the registry."""

    def register(criterion):
        CRITERIA[name] = criterion
        return criterion

    return register


def get_criterion(config, modules):
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
        """Initiliaze a pruning criterion."""
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


@register_criterion('magnitude')
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
        """Initiliaze a magnitude pruning criterion."""
        super(MagnitudeCriterion, self).__init__(modules, config)

    def on_step_begin(self):
        """Calculate and store the pruning scores based on a magnitude criterion."""
        with torch.no_grad():
            for key in self.modules.keys():
                p = self.modules[key].weight.data
                self.scores[key] = torch.abs(p)


@register_criterion('gradient')
class GradientCriterion(PruningCriterion):
    """Pruning criterion.
    
    The gradient criterion_class is derived from PruningCriterion.
    The absolute value of gradient is used to score and determine if a weight is to be pruned.

    Args:
        config: A config dict object that includes information about pruner and pruning criterion.
        modules: A dict {"module_name": Tensor} that stores the pruning modules' weights.
    
    Attributes:
        scores: A dict {"module_name": Tensor} that stores the scores of pruning modules.
    """

    def __init__(self, modules, config):
        """Initiliaze a gradient pruning criterion."""
        super(GradientCriterion, self).__init__(modules, config)
        assert self.config.end_step > 0, "please set end_step > 0 for gradient based criterion"

    def on_before_optimizer_step(self):
        """Calculate and store the pruning scores based on gradient criterion."""
        with torch.no_grad():
            for key in self.modules.keys():
                p = self.modules[key].weight
                self.scores[key] = torch.abs(p.grad)


@register_criterion('snip')
class SnipCriterion(PruningCriterion):
    """Pruning criterion.
    
    The snip criterion_class is derived from PruningCriterion.
    The product of magnitude and gradient is used to score and determine if a weight is to be pruned.
    Please refer to SNIP: Single-shot Network Pruning based on Connection Sensitivity.
    (https://arxiv.org/abs/1810.02340)

    Args:
        config: A config dict object that includes information about pruner and pruning criterion.
        modules: A dict {"module_name": Tensor} that stores the pruning modules' weights.
    
    Attributes:
        scores: A dict {"module_name": Tensor} that stores the scores of pruning modules.
    """

    def __init__(self, modules, config):
        """Initiliaze a snip pruning criterion."""
        super(SnipCriterion, self).__init__(modules, config)
        assert self.config.end_step > 0, "please set end_step > 0 for gradient based criterion"

    def on_before_optimizer_step(self):
        """Calculate and store the pruning scores based on snip criterion."""
        ##self.mask_weights()
        with torch.no_grad():
            for key in self.modules.keys():
                p = self.modules[key].weight
                self.scores[key] = torch.abs(p * p.grad)


@register_criterion('snip_momentum')
class SnipMomentumCriterion(PruningCriterion):
    """Pruning criterion.
    
    The snip_momentum criterion_class is derived from PruningCriterion.
    A momentum mechanism is used to calculate snip score, which determines if a weight is to be pruned.

    Args:
        config: A config dict object that includes information about pruner and pruning criterion.
        modules: A dict {"module_name": Tensor} that stores the pruning modules' weights.
        alpha: A parameter that determines how much of the snip score is preserved from last pruning step.
        beta: A parameter that determines how much of the snip score is updated at the current step.
    
    Attributes:
        scores: A dict {"module_name": Tensor} that stores the scores of pruning modules.
    """

    def __init__(self, modules, config):
        """Initiliaze a snip_momentum pruning criterion."""
        super(SnipMomentumCriterion, self).__init__(modules, config)
        assert self.config.end_step > 0, "please set end_step > 0 for gradient based criterion"
        for key in modules.keys():
            p = modules[key].weight
            self.scores[key] = torch.zeros(p.shape).to(p.device)

        self.alpha = 0.9
        self.beta = 1.0

    def on_before_optimizer_step(self):
        """Calculate and store the pruning scores based on snip_momentum criterion."""
        with torch.no_grad():
            for key in self.modules.keys():
                p = self.modules[key].weight
                self.scores[key] *= self.alpha
                self.scores[key] += self.beta * torch.abs(p * p.grad)
                
                
@register_criterion('snip_momentum_block')
class SnipMomentumBlockCriterion(PruningCriterion):
    """Pruning criterion.
    
    The snip_momentum_block criterion_class is derived from PruningCriterion.
    A momentum mechanism is used to calculate snip score, which determines if a block weight is to be pruned.

    Args:
        config: A config dict object that includes information about pruner and pruning criterion.
        modules: A dict {"module_name": Tensor} that stores the pruning modules' weights.
        alpha: A parameter that determines how much of the snip score is preserved from last pruning step.
        beta: A parameter that determines how much of the snip score is updated at the current step.
    
    Attributes:
        scores: A dict {"module_name": Tensor} that stores the scores of pruning modules.
    """

    def __init__(self, modules, config):
        """Initiliaze a block_mask pruning criterion."""
        super(SnipMomentumBlockCriterion, self).__init__(modules, config)
        assert self.config.end_step > 0, "please set end_step > 0 for gradient based criterion"
        for key in self.modules.keys():
            if not hasattr(self.modules[key], 'block_mask'):
                continue # No corresponding block mask, skip.
            mask = self.modules[key].block_mask
            self.scores[key] = torch.zeros(mask.shape).to(mask.device)
        self.alpha = 0.9
        self.beta = 1.0
        
    def on_train_begin(self):
        """Initiliaze the block shape scores."""
        pass

    def on_before_optimizer_step(self):
        """Calculate and store the pruning scores based on snip_momentum_block criterion."""
        with torch.no_grad():
            for key in self.modules.keys():
                if not hasattr(self.modules[key], 'block_mask'):
                    continue # No corresponding block mask, skip.
                mask = self.modules[key].block_mask
                self.scores[key] *= self.alpha
                reduce_weight = self.reduce_weights(self.modules[key])
                self.scores[key] += self.beta * torch.abs(reduce_weight * mask.grad)
                
    def reduce_weights(self, module):
        """Calculate the reduced weights by block."""
        if type(module).__name__ not in ["Linear"]: # Currently only linear is supported
            return module.block_mask
        block_size = [module.weight.shape[0]//module.block_mask.shape[0], \
                      module.weight.shape[1]//module.block_mask.shape[1]] 
        shape = module.weight.shape
        new_shape = [shape[0] // block_size[0], block_size[0], shape[1] // block_size[1],
                     block_size[1]]
        weight = module.weight.data.reshape(new_shape)
        reduced_weight = self.reduce_tensor(self.reduce_tensor(weight, dim=-1), dim=1)
        return reduced_weight
        
    def reduce_tensor(self, data, dim):
        """Reduce the data along the given dimension."""
        name = self.config['criterion_reduce_type']
        if name == "mean":
            return torch.mean(data, dim=dim)
        elif name == "sum":
            return torch.sum(data, dim=dim)
        elif name == "max":
            return torch.max(data, dim=dim)[0]
        else:
            assert False, "currently only support mean, sum and max reduce type"
