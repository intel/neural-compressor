"""Basic pruner."""

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

# from ..utils import logger
from neural_compressor.utils.logger import Logger

from ..criteria import get_criterion
from ..patterns import get_pattern
from ..regs import get_reg
from ..schedulers import get_scheduler
from ..tf_criteria import get_tf_criterion
from .base import KerasBasePruner, PytorchBasePruner, register_pruner

logger = Logger().get_logger()


@register_pruner("pt_basic")
class PytorchBasicPruner(PytorchBasePruner):
    """Pruning Pruner.

    The class which executes pruning process.
    1. Defines pruning functions called at step begin/end, epoch begin/end.
    2. Defines the pruning criterion.

    Args:
        modules: A dict {"module_name": Tensor} that stores the pruning modules' weights.
        config: A config dict object that contains the pruner information.

    Attributes:
        pattern: A Pattern object that defines pruning weights' arrangements within space.
        criterion: A Criterion Object that defines which weights are to be pruned
        scheduler: A Scheduler object that defines how the model's sparsity changes as training/pruning proceeds.
        reg: A Reg object that defines regulization terms.
    """

    def __init__(self, config, modules):
        """Initialize."""
        super().__init__(config, modules)

    def _init(self):
        """Auxiliary function for initializing."""
        self.pattern = get_pattern(self.config, self.modules)
        self.scheduler = get_scheduler(self.config)
        self.criterion = get_criterion(self.config, self.modules, self.pattern)
        self.reg = get_reg(self.config, self.modules, self.pattern)
        # if switch off progressive but use per-channel pruning, give a warn
        if "channel" in self.pattern.pattern:
            logger.info("UserWarning: use per-channel pruning pattern without progressive pruning!")
            logger.info("Instead, enabling progressive pruning would be a better choice.")
        else:
            pass

    def set_global_step(self, global_step):
        """Set global step number."""
        self.global_step = global_step

    # def on_step_begin(self, local_step):
    #     """Implement at the start of each step.
    #
    #     Update the masks at a given local_step.
    #     """
    #     self.update_masks(local_step)

    def update_masks(self, local_step):
        """Update the masks at a given local step."""
        if self.global_step == self.start_step:
            if self.config["lock_init_sparsity"]:
                self.masks = self.pattern.get_pattern_lock_masks(self.modules)
                self.init_sparsity_ratio = self.pattern.get_sparsity_ratio(self.masks)
                self.current_sparsity_ratio = self.init_sparsity_ratio

        if not self.check_is_pruned_step(self.global_step):
            return

        if self.current_sparsity_ratio > self.target_sparsity_ratio:
            return

        self.criterion.on_step_begin()
        current_target_sparsity_ratio = self.scheduler.update_sparsity_ratio(
            self.target_sparsity_ratio,
            self.completed_pruned_cnt,
            self.total_prune_cnt,
            self.masks,
            self.init_sparsity_ratio,
        )
        logger.info(f"current target ratio is {current_target_sparsity_ratio}")

        self.completed_pruned_cnt += 1
        if self.criterion.scores == {}:
            return
        self.masks = self.pattern.get_masks(self.criterion.scores, current_target_sparsity_ratio, self.masks)
        self.mask_weights()

        self.current_sparsity_ratio = self.pattern.get_sparsity_ratio(self.masks)
        logger.info(f"current sparsity ratio is {self.current_sparsity_ratio}")

    def on_before_optimizer_step(self):
        """Implement before optimizer.step()."""
        self.reg.on_before_optimizer_step()
        self.criterion.on_before_optimizer_step()

    def on_after_optimizer_step(self):
        """Prune the model after optimization."""
        # the order of the following three lines can't not be exchanged
        if self.global_step >= self.start_step and self.global_step <= self.end_step:
            self.reg.on_after_optimizer_step()
        self.mask_weights()

        self.global_step += 1


@register_pruner("keras_basic")
class KerasBasicPruner(KerasBasePruner):
    """Pruning Pruner.

    The class which executes pruning process.
    1. Defines pruning functions called at step begin/end, epoch begin/end.
    2. Defines the pruning criterion.

    Args:
        modules: A dict {"module_name": Tensor} that stores the pruning modules' weights.
        config: A config dict object that contains the pruner information.

    Attributes:
        pattern: A Pattern object that defines pruning weights' arrangements within space.
        criterion: A Criterion Object that defines which weights are to be pruned
        scheduler: A Scheduler object that defines how the model's sparsity changes as training/pruning proceeds.
        reg: A Reg object that defines regulization terms.
    """

    def _init(self):
        """Auxiliary function for initializing."""
        self.pattern = get_pattern(self.config, self.modules, framework="keras")
        self.scheduler = get_scheduler(self.config)
        self.criterion = get_tf_criterion(self.config, self.modules)
        self.reg = get_reg(self.config, self.modules, self.pattern)
        # if switch off progressive but use per-channel pruning, give a warn
        if "channel" in self.pattern.pattern:
            logger.info("UserWarning: use per-channel pruning pattern without progressive pruning!")
            logger.info("Instead, enabling progressive pruning would be a better choice.")
        else:
            pass

    def set_global_step(self, global_step):
        """Set global step number."""
        self.global_step = global_step

    # def on_step_begin(self, local_step):
    #     """Implement at the start of each step.
    #
    #     Update the masks at a given local_step.
    #     """
    #     self.update_masks(local_step)

    def update_masks(self, local_step):
        """Update the masks at a given local step."""
        if self.global_step == self.start_step:
            if self.config["lock_init_sparsity"]:
                self.masks = self.pattern.get_pattern_lock_masks(self.modules)
                self.init_sparsity_ratio = self.pattern.get_sparsity_ratio(self.masks)
                self.current_sparsity_ratio = self.init_sparsity_ratio

        if not self.check_is_pruned_step(self.global_step):
            return

        if self.current_sparsity_ratio > self.target_sparsity_ratio:
            return

        self.criterion.on_step_begin()
        current_target_sparsity_ratio = self.scheduler.update_sparsity_ratio(
            self.target_sparsity_ratio,
            self.completed_pruned_cnt,
            self.total_prune_cnt,
            self.masks,
            self.init_sparsity_ratio,
        )
        logger.info(f"current target ratio is {current_target_sparsity_ratio}")

        self.completed_pruned_cnt += 1
        if self.criterion.scores == {}:
            return
        self.masks = self.pattern.get_masks(self.criterion.scores, current_target_sparsity_ratio, self.masks)
        self.mask_weights()

        self.current_sparsity_ratio = self.pattern.get_sparsity_ratio(self.masks)
        logger.info(f"current sparsity ratio is {self.current_sparsity_ratio}")

    def on_before_optimizer_step(self):
        """Implement before optimizer.step()."""
        self.reg.on_before_optimizer_step()
        self.criterion.on_before_optimizer_step()

    def on_after_optimizer_step(self):
        """Prune the model after optimization."""
        # the order of the following three lines can't not be exchanged
        if self.global_step >= self.start_step and self.global_step <= self.end_step:
            self.reg.on_after_optimizer_step()
        self.mask_weights()

        self.global_step += 1
