"""scheduler module."""
#!/usr/bin/env python
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

import math

SCHEDULERS = {}


def register_scheduler(name):
    """Class decorator used to register a Scheduler subclass to the registry.

    Decorator function used before a Scheduler subclass.
    Make sure that the Scheduler class decorated by this function can be registered in SCHEDULERS.
    
    Args:
        cls (class): The class of register.
        name: A string. Define the scheduler type.

    Returns:
        cls: The class of register.
    """

    def register(scheduler):
        SCHEDULERS[name] = scheduler
        return scheduler

    return register


def get_scheduler(config):
    """Get registered scheduler class.

    Get a scheduler object from SCHEDULERS.

    Args:
        config: A config dict object. Contains the scheduler information.

    Returns:
        A Scheduler object.
    """
    name = "iterative"
    if config.start_step == config.end_step:
        name = "oneshot"
    return SCHEDULERS[name](config)


class Scheduler:
    """Pruning Scheduler.

    The class which defines a sparsity changing process during pruning.
    Mainly contains two types:
        1. iterative scheduler. Prune the model from dense to target sparsity gradually.
        2. one-shot scheduler. Prune the model in a single step and reach the target sparsity.

    Args:
        config: A config dict object. Contains the scheduler information.

    Attributes:
        config: A config dict object. Contains the scheduler information.
    """

    def __init__(self, config):
        """Initialize."""
        self.config = config

    def update_sparsity_ratio(self, aggressive_ratio, current_prune_step, total_prune_steps, masks):
        """To be implemented in subclasses."""
        raise NotImplementedError


@register_scheduler('oneshot')
class OneshotScheduler(Scheduler):
    """Pruning Scheduler.

    A Scheduler class derived from Scheduler.
    Prune the model to target sparsity once.

    Args:
        config: A config dict object. Contains the scheduler information.

    Attributes:
        Inherit from parent class Scheduler.
    """

    def __init__(self, config):
        """Initialize."""
        super(OneshotScheduler, self).__init__(config)

    def update_sparsity_ratio(self, aggressive_ratio, current_prune_step, total_prune_steps, masks):
        """Return the aggressive ratio."""
        return aggressive_ratio


@register_scheduler('iterative')
class IterativeScheduler(Scheduler):
    """Pruning Scheduler.

    A Scheduler class derived from Scheduler.
    Prune the model to from dense to target sparsity in several steps.

    Args:
        config: A config dict object. Contains the scheduler information.

    Attributes:
        Inherit from parent class Scheduler.
    """

    def __init__(self, config):
        """Initialize."""
        super(IterativeScheduler, self).__init__(config)
        # self.decay_type = config["sparsity_decay_type"]

    def update_sparsity_ratio(self, target_ratio, current_prune_step, total_prune_steps, masks):
        """Obtain new target sparsity ratio according to the step.

        Args:
            target_ratio: A float. The target sparsity ratio.
            current_prune_step: A integer. The current pruning step.
            total_prune_steps: A integer. The total steps included in the pruning progress.
            masks: A dict{"module_name": Tensor}. The masks for modules' weights.
        
        Returns：
            A float. the target sparsity ratio the model will reach after the next pruning step.
        """
        aggressive_ratio = target_ratio
        # if self.config.prune_domain == "global":
        #     aggressive_ratio += 0.02

        aggressive_ratio = min(self.config.max_sparsity_ratio_per_layer,
                               aggressive_ratio)  ##lagacy issue

        decay_type = self.config.sparsity_decay_type
        if decay_type == "cos":
            current_target_sparsity = (aggressive_ratio) * (
                    1.0 - math.cos(float(current_prune_step) / total_prune_steps * (math.pi / 2)))
        elif decay_type == "exp":
            target_dense_change_ratio = (1.0 - aggressive_ratio) ** (1 / total_prune_steps)
            current_target_sparsity = 1.0 - target_dense_change_ratio ** current_prune_step

        elif decay_type == "linear":
            current_target_sparsity = (aggressive_ratio) * float(current_prune_step) / total_prune_steps

        elif decay_type == "cube":
            current_target_sparsity = (aggressive_ratio) * ((float(current_prune_step) / total_prune_steps) ** 3)
        else:
            assert False, "{} is not supported".format(decay_type)

        current_target_sparsity = min(target_ratio, current_target_sparsity)
        return current_target_sparsity
