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

import torch
from .patterns import get_pattern
from .scheduler import get_scheduler

from .logger import logger

PRUNERS = {}


def register_pruners(name):
    """Register a pruner to the registry"""

    def register(pruner):
        PRUNERS[name] = pruner
        return pruner

    return register


def get_pruner(moduels, config):
    """Get registered pruner class"""
    name = config["prune_type"]
    if name not in PRUNERS.keys():
        assert False, f"does not support {name}, currently only support {PRUNERS.keys()}"
    return PRUNERS[name](moduels, config)


class Pruner:
    def __init__(self, modules, config):
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
        pass

    def mask_weights(self):
        with torch.no_grad():
            for key in self.modules.keys():
                module = self.modules[key]
                module.weight.data = module.weight.data * self.masks[key]

    def on_step_begin(self, local_step):
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

    def on_epoch_end(self):
        pass

    def on_step_end(self):
        pass

    def on_before_optimizer_step(self):
        pass

    def on_after_optimizer_step(self):
        self.mask_weights()

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

    def check_is_pruned_step(self, step):
        if step < self.start_step or step > self.end_step:
            return False
        if int(step - self.start_step) % self.update_frequency_on_step == 0:
            return True
        return False

    def update_scores(self):
        pass


@register_pruners('snip')
class SnipPruner(Pruner):
    """
    please refer to SNIP: Single-shot Network Pruning based on Connection Sensitivity 
    (https://arxiv.org/abs/1810.02340)
    """
    def __init__(self, modules, config):
        super(SnipPruner, self).__init__(modules, config)
        assert self.config.end_step > 0, "gradient based criteria does not work on step 0"
        self.scores = {}

    def on_after_optimizer_step(self):
        with torch.no_grad():
            for key in self.modules.keys():
                p = self.modules[key].weight
                self.scores[key] = torch.abs(p * p.grad)
        self.mask_weights()


@register_pruners('snip_momentum')
class SnipMomentumPruner(Pruner):
    def __init__(self, modules, config):
        super(SnipMomentumPruner, self).__init__(modules, config)
        assert self.config.end_step > 0, "gradient based criteria does not work on step 0"
        # self.scores = {}
        for key in modules.keys():
            p = modules[key].weight
            self.scores[key] = torch.zeros(p.shape).to(p.device)

    def on_after_optimizer_step(self):
        with torch.no_grad():
            for key in self.modules.keys():
                p = self.modules[key].weight
                self.scores[key] *= 0.9  ##magic number
                self.scores[key] += 1.0 * torch.abs(p * p.grad)
        self.mask_weights()


@register_pruners('magnitude')
class MagnitudePruner(Pruner):
    def __init__(self, modules, config):
        super(MagnitudePruner, self).__init__(modules, config)
        self.scores = {}

    def update_scores(self):
        with torch.no_grad():
            for key in self.modules.keys():
                p = self.modules[key].weight.data
                self.scores[key] = p
