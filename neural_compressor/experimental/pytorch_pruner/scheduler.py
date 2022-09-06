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
    """Register a scheduler to the registry"""

    def register(scheduler):
        SCHEDULERS[name] = scheduler
        return scheduler

    return register


def get_scheduler(config):
    """Get registered scheduler class"""
    name = "iterative"
    if config.start_step == config.end_step:
        name = "oneshot"
    return SCHEDULERS[name](config)


class Scheduler:
    def __init__(self, config):
        self.config = config

    def update_sparsity_ratio(self, aggressive_ratio, current_prune_step, total_prune_steps, masks):
        raise NotImplementedError


@register_scheduler('oneshot')
class OneshotScheduler(Scheduler):
    def __init__(self, config):
        super(OneshotScheduler, self).__init__(config)

    def update_sparsity_ratio(self, aggressive_ratio, current_prune_step, total_prune_steps, masks):
        return aggressive_ratio


@register_scheduler('iterative')
class IterativeScheduler(Scheduler):
    def __init__(self, config):
        super(IterativeScheduler, self).__init__(config)
        # self.decay_type = config["sparsity_decay_type"]

    def update_sparsity_ratio(self, target_ratio, current_prune_step, total_prune_steps, masks):
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
