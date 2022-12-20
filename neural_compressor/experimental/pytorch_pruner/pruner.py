"""pruner module."""
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
from .patterns import get_pattern
from .scheduler import get_scheduler

from .logger import logger

PRUNERS = {}


def register_pruners(name):
    """Class decorator to register a Pruner subclass to the registry.

    Decorator function used before a Pattern subclass.
    Make sure that the Pruner class decorated by this function can be registered in PRUNERS.

    Args:
        cls (class): The subclass of register.
        name: A string. Define the pruner type.

    Returns:
        cls: The class of register.
    """

    def register(pruner):
        PRUNERS[name] = pruner
        return pruner

    return register


def get_pruner(modules, config):
    """Get registered pruner class.

    Get a Pruner object from PRUNERS.
    
    Args:
        modules: A dict {"module_name": Tensor}. Store the pruning modules' weights.
        config: A config dict object. Contains the pruner information.

    Returns:
        A Pruner object.
    
    Raises: AssertionError: Cuurently only support pruners which have been registered in PRUNERS.
    """
    name = config["prune_type"]
    if name not in PRUNERS.keys():
        assert False, f"does not support {name}, currently only support {PRUNERS.keys()}"
    return PRUNERS[name](modules, config)


class Pruner:
    """Pruning Pruner.

    The class which executes pruning process.
    1. Defines pruning functions called at step begin/end, epoch begin/end.
    2. Defines the pruning criteria.

    Args:
        modules: A dict {"module_name": Tensor}. Store the pruning modules' weights.
        config: A config dict object. Contains the pruner information.

    Attributes:
        modules: A dict {"module_name": Tensor}. Store the pruning modules' weights.
        config: A config dict object. Contains the pruner information.
        masks: A dict {"module_name": Tensor}. Store the masks for modules' weights.
        scores: A dict {"module_name": Tensor}. Store the score for modules' weights,
            which are used to decide pruning parts with a criteria.
        pattern: A Pattern object. Defined in ./patterns.py
        scheduler: A scheduler object. Defined in ./scheduler.py
        current_sparsity_ratio: A float. Current model's sparsity ratio, initialized as zero.
        global_step: A integer. The total steps the model has run.
        start_step: A integer. When to trigger pruning process.
        end_step: A integer. When to end pruning process.
        update_frequency_on_step: A integer. The pruning frequency, which's valid when iterative 
            pruning is enabled.
        target_sparsity_ratio: A float. The final sparsity after pruning.
        max_sparsity_ratio_per_layer: A float. Sparsity ratio maximum for every module.
    """

    def __init__(self, modules, config):
        """Initialize."""
        self.modules = modules
        self.config = config
        self.masks = {}
        self.scores = {}
        self.reg = None  ##TODO need to add reg
        self.pattern = get_pattern(config)
        self.scheduler = get_scheduler(config)
        self.current_sparsity_ratio = 0.0
        self._init()

    def _init(self):
        """Auxiliary function for initializing."""
        self.global_step = -1
        self.start_step = self.config['start_step']
        self.end_step = self.config['end_step']
        self.update_frequency_on_step = self.config['update_frequency_on_step']
        ##this is different with original code
        self.total_prune_cnt = (self.end_step - self.start_step + 1) \
                               // self.update_frequency_on_step
        self.completed_pruned_cnt = 0
        self.masks = {}
        for key in self.modules.keys():
            module = self.modules[key]
            self.masks[key] = torch.ones(module.weight.shape).to(module.weight.device)  ##TODO support bias or others

        self.target_sparsity_ratio = self.config['target_sparsity']

        self.max_sparsity_ratio_per_layer = self.config['max_sparsity_ratio_per_layer']

    def on_epoch_begin(self, epoch):
        """Functions called in the beginning of each epoch."""
        pass

    def mask_weights(self):
        """Functions called when masks are applied on corresponding modules' weights.

        Weights are multipled with masks. This is the formal pruning process.
        """
        with torch.no_grad():
            for key in self.modules.keys():
                module = self.modules[key]
                module.weight.data = module.weight.data * self.masks[key]

    def on_step_begin(self, local_step):
        """Functions called on the beginning of each step.

        Judge if the current step should execute a pruning process.
        If so, using scores and criteria to update the masks and pruning the model.
        Or, simply train the model with its original structure.
        """
        self.global_step += 1
        if not self.check_is_pruned_step(self.global_step):
            return

        if self.current_sparsity_ratio > self.target_sparsity_ratio:
            return

        current_target_sparsity_ratio = self.scheduler.update_sparsity_ratio(self.target_sparsity_ratio,
                                                                             self.completed_pruned_cnt,
                                                                             self.total_prune_cnt, self.masks)
        logger.info(f"current target ratio is {current_target_sparsity_ratio}")
        self.update_scores()
        self.completed_pruned_cnt += 1
        if self.scores == {}:
            return
        self.masks = self.pattern.get_masks(self.scores, current_target_sparsity_ratio, self.masks,
                                            self.max_sparsity_ratio_per_layer)
        self.mask_weights()

        self.current_sparsity_ratio = self.pattern.get_sparsity_ratio(self.masks)
        logger.info(f"current sparsity ratio is {self.current_sparsity_ratio}")

    def on_step_end(self):
        """Functions called in the end of each step."""
        pass

    def on_epoch_end(self):
        """Functions called in the end of each epoch."""
        pass

    def on_before_optimizer_step(self):
        """Functions called before the optimizer.step()."""
        pass

    def on_after_optimizer_step(self):
        """Functions called after the optimizer.step().
        
        Prune the model after optimization.
        """
        self.mask_weights()

    def on_train_begin(self, dataloader = None):
        """Functions called in the beginning of training."""
        pass

    def on_train_end(self):
        """Functions called in the end of each training."""
        pass

    def on_before_eval(self):
        """Functions called in the beginning of evaluation."""
        pass

    def on_after_eval(self):
        """Functions called in the end of evaluation."""
        pass

    def check_is_pruned_step(self, step):
        """Decide whether the current step should execute a pruning process."""
        if step < self.start_step or step > self.end_step:
            return False
        if int(step - self.start_step) % self.update_frequency_on_step == 0:
            return True
        return False

    def update_scores(self):
        """Update self.scores."""
        pass


@register_pruners('magnitude')
class MagnitudePruner(Pruner):
    """Pruning Pruner.

    A Pruner class derived from Pruner. In this pruner, the scores are calculated based on weights.

    Args:
        modules: A dict {"module_name": Tensor}. Store the pruning modules' weights.
        config: A config dict object. Contains the pruner information.

    Attributes:
        Inherit from parent class Pruner.
    """

    def __init__(self, modules, config):
        """Initialize."""
        super(MagnitudePruner, self).__init__(modules, config)
        self.scores = {}

    def update_scores(self):
        """Update self.scores."""
        with torch.no_grad():
            for key in self.modules.keys():
                p = self.modules[key].weight.data
                self.scores[key] = p


@register_pruners('snip')
class SnipPruner(Pruner):
    """Pruning Pruner.

    A Pruner class derived from Pruner. In this pruner, the scores are calculated based on SNIP.
    Please refer to SNIP: Single-shot Network Pruning based on Connection Sensitivity 
    (https://arxiv.org/abs/1810.02340)

    Args:
        modules: A dict {"module_name": Tensor}. Store the pruning modules' weights.
        config: A config dict object. Contains the pruner information.

    Attributes:
        Inherit from parent class Pruner.
    """

    def __init__(self, modules, config):
        """Initialize."""
        super(SnipPruner, self).__init__(modules, config)
        assert self.config.end_step > 0, "gradient based criteria does not work on step 0"
        self.scores = {}

    def on_after_optimizer_step(self):
        """Functions called after the optimizer.step().
        
        Prune the model after optimization and update the scores based on weights and gradients.
        """
        self.mask_weights()
        with torch.no_grad():
            for key in self.modules.keys():
                p = self.modules[key].weight
                self.scores[key] = torch.abs(p * p.grad)


@register_pruners('snip_momentum')
class SnipMomentumPruner(Pruner):
    """Pruning Pruner.

    A Pruner class derived from Pruner. In this pruner, the scores are calculated based on SNIP.
    Moreoever, the score map is updated with a momentum like process.

    Args:
        modules: A dict {"module_name": Tensor}. Store the pruning modules' weights.
        config: A config dict object. Contains the pruner information.

    Attributes:
        Inherit from parent class Pruner.
    """

    def __init__(self, modules, config):
        """Initialize."""
        super(SnipMomentumPruner, self).__init__(modules, config)
        assert self.config.end_step > 0, "gradient based criteria does not work on step 0"
        # self.scores = {}
        for key in modules.keys():
            p = modules[key].weight
            self.scores[key] = torch.zeros(p.shape).to(p.device)

    def on_after_optimizer_step(self):
        """Functions called after the optimizer.step().
        
        Prune the model after optimization and update the scores based on weights and gradients.
        """
        self.mask_weights()
        with torch.no_grad():
            for key in self.modules.keys():
                p = self.modules[key].weight
                self.scores[key] *= 0.9  ##magic number
                self.scores[key] += 1.0 * torch.abs(p * p.grad)


@register_pruners('pattern_lock')
class PatternLockPruner(Pruner):
    """Pruning Pruner.

    A Pruner class derived from Pruner. In this pruner, original model's sparsity pattern will be fixed while training.
    This pruner is useful when you want to train a sparse model without change its original structure.

    Args:
        modules: A dict {"module_name": Tensor}. Store the pruning modules' weights.
        config: A config dict object. Contains the pruner information.

    Attributes:
        Inherit from parent class Pruner.
    """

    def __init__(self, modules, config):
        """Initialize."""
        super(PatternLockPruner, self).__init__(modules, config)
        assert self.config.end_step == self.config.start_step, "pattern_lock pruner only supports one shot mode"

    def on_step_begin(self, local_step):
        """Functions called on the beginning of each step."""
        self.global_step += 1
        if not self.check_is_pruned_step(self.global_step):
            return
        self.masks = self.pattern.get_pattern_lock_masks(self.modules)

    def on_after_optimizer_step(self):
        """Functions called after the optimizer.step()."""
        self.mask_weights()
