"""Progressive pruner."""

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

import copy

from ..criteria import get_criterion
from ..patterns import get_pattern
from ..regs import get_reg
from ..schedulers import get_scheduler
from ..utils import logger, torch
from .base import register_pruner
from .basic import PytorchBasicPruner


@register_pruner("pt_progressive")
class PytorchProgressivePruner(PytorchBasicPruner):
    """Pruning Pruner.

    A Pruner class derived from BasicPruner. In this pruner, mask interpolation will be applied.
    Mask interpolation is a fine-grained improvement for NxM structured pruning by adding interval
        masks between masks of two pruning steps.

    Args:
        modules: A dict {"module_name": Tensor} that stores the pruning modules' weights.
        config: A config dict object that contains the pruner information.

    Attributes:
        Inherit from parent class Pruner.
    """

    def __init__(self, config, modules):
        """Initialize."""
        super().__init__(config, modules)

    def _init(self):
        """Auxiliary function for initialization."""
        self.pattern = get_pattern(self.config, self.modules)
        self.scheduler = get_scheduler(self.config)
        self.criterion = get_criterion(self.config, self.modules, self.pattern)
        self.reg = get_reg(self.config, self.modules, self.pattern)
        # progressive pruning set up, including check up parameters.
        self.use_progressive = self.config["progressive"]
        # progressive parameters
        # dict passed to Pattern's functions
        self.progressive_configs = {"progressive_steps": 4, "progressive_type": "scores", "use_global": True}
        self.progressive_steps = self.progressive_configs["progressive_steps"]
        self.progressive_type = self.progressive_configs["progressive_type"]
        self.use_global = self.progressive_configs["use_global"]
        self.progressive_logger = True
        self.align_masks_flag = False
        self._init_for_progressive()

    def _init_for_progressive(self):
        """Auxiliary function for initializing progressive pruning."""
        # detailed progressive parameters will stored at patterns.py
        # step 1: check if pattern is NxM
        # if "x" not in self.pattern.pattern:
        #     raise NotImplementedError(f"Currently progressive only " \
        #                               f"support NxM and per-channel pruning patterns.")

        # step 2: check if current set up will "degrade" into non-progressive
        if (self.end_step - self.start_step) <= self.progressive_steps or self.progressive_steps <= 1:
            logger.info("Current progressive setting will degrading to non-progressive pruning.")
            self.use_progressive = False
            return

        if self.pruning_frequency == 1:
            logger.info("Current progressive setting will degrading to non-progressive pruning.")
            self.use_progressive = False
            return

        # step 3: log hyper-parameters. and check validity.
        if self.use_progressive:
            logger.info("Progressive pruning is enabled!")
            logger.info(f"Progressive pruning steps: {self.progressive_steps}")
            logger.info(f"Progressive type: {self.progressive_type}")
            logger.info(f"Progressive balance: {self.use_global}")
            self.check_progressive_validity()
            self.pre_masks = copy.deepcopy(self.masks)
            self.progressive_masks = copy.deepcopy(self.masks)
            if self.pruning_frequency < self.progressive_steps:  # TODO: trick
                self.progressive_steps = self.pruning_frequency
                # if self.progressive_steps == 3:
                #     self.progressive_steps = 2
                self.pruning_frequency_progressive = self.progressive_steps
            else:
                self.pruning_frequency_progressive = self.pruning_frequency // self.progressive_steps
            # this is a structural pruning step, it fits self.pruning_frequency
            self.structured_update_step = 0

    def check_progressive_validity(self):
        """Check if the settings of progressive pruning are valid."""
        # check some problematic settings
        if self.progressive_type == "linear":
            # linear based progressive pruning, only valid for NxM pattern
            assert type(self.pattern).__name__ == "PytorchPatternNxM", "Progressive linear pruning only support NxM."
            if self.use_global:
                # when global progressive is applied, linear type is contradict.
                raise NotImplementedError("Global progressive pruning do not support linear pattern")
            # When linear, progressive_step should not meet a indivisible
            for key in self.pattern.block_size.keys():
                block_size = self.pattern.block_size[key]
                progressive_direction = max(block_size)
                if progressive_direction % self.progressive_steps != 0:
                    raise ValueError(
                        f"In layer {key}, its pruning pattern is {block_size}, "
                        f"while progressive steps {self.progressive_steps} is indivisible."
                    )
        else:
            # score based progressive pruning, support both NxM and N:M patterns
            if type(self.pattern).__name__ == "PytorchPatternNxM":
                for key in self.pattern.block_size.keys():
                    block_size = self.pattern.block_size[key]
                    total_block_size = block_size[0] * block_size[1]
                    if total_block_size < self.progressive_steps:
                        raise ValueError(
                            f"In layer {key}, its pruning pattern is {block_size}, "
                            f"while progressive steps {self.progressive_steps} is overflowing."
                        )
            elif type(self.pattern).__name__ == "PytorchPatternNInM":
                if self.pattern.N < self.progressive_steps:
                    raise ValueError(
                        f"Pruning pattern is {self.pattern.N} in {self.pattern.M}, "
                        f"while progressive steps {self.progressive_steps} is overflowing."
                    )
            else:
                raise NotImplementedError

    def check_is_pruned_progressive_step(self, step):
        """Check if a progressive pruning process should be performed at the current step.

        Args:
            step: an integer representing the number of current step.

        Returns:
            A Boolean.
        """
        # used in progressive pruning
        if step < self.start_step or step > self.end_step:
            return False
        if int(step - self.start_step) % self.pruning_frequency_progressive == 0:
            return True
        return False

    def update_masks_progressive(self, local_step):
        """Update the masks in progressive pruning mode at a given local step."""
        if self.global_step == self.start_step:
            if self.config["lock_init_sparsity"]:
                self.masks = self.pattern.get_pattern_lock_masks(self.modules)
                self.init_sparsity_ratio = self.pattern.get_sparsity_ratio(self.masks)
                self.current_sparsity_ratio = self.init_sparsity_ratio

        # case 1: step is not in [start_step, end_step] or it is not either pruning or progressive pruning step.
        if (self.check_is_pruned_step(self.global_step) is False) and (
            self.check_is_pruned_progressive_step(self.global_step) is False
        ):
            return
        if self.current_sparsity_ratio > self.target_sparsity_ratio:
            return

        # case 2: step which does progressive update, but it is not a pruning step in case 3
        if (
            self.check_is_pruned_progressive_step(self.global_step)
            and self.check_is_pruned_step(self.global_step) is False
        ):
            # do not do global pruning, only do the progressive mask update.
            step_offset = self.global_step - self.structured_update_step
            progressive_idx = step_offset // self.pruning_frequency_progressive
            if progressive_idx < (self.progressive_steps - 1):
                self.progressive_masks = self.pattern.update_progressive_masks(
                    self.pre_masks, self.masks, self.criterion.scores, progressive_idx + 1, self.progressive_configs
                )
            else:
                # in the end, directly use new masks.
                for n in self.masks.keys():
                    self.progressive_masks[n] = self.masks[n].clone()
            self.mask_weights_general(self.progressive_masks)
            if self.progressive_logger:
                self.print_progressive_sparsity()
            return

        # case 3: a pruning step, generate new masks, progressive masks also update.
        tmp_step = self.global_step
        self.structured_update_step = tmp_step
        current_target_sparsity_ratio = self.scheduler.update_sparsity_ratio(
            self.target_sparsity_ratio, self.completed_pruned_cnt, self.total_prune_cnt, self.masks
        )
        logger.info(f"current target ratio is {current_target_sparsity_ratio}")
        self.criterion.on_step_begin()
        self.completed_pruned_cnt += 1
        if self.criterion.scores == {}:
            return
        for n in self.masks.keys():
            self.pre_masks[n] = self.masks[n].clone()
        # update new masks
        # if not self.use_progressive:
        #     self.masks = self.pattern.get_masks(
        #         self.criterion.scores,
        #         current_target_sparsity_ratio,
        #         self.masks,
        #     )
        self.masks = self.pattern.get_masks(
            self.criterion.scores,
            current_target_sparsity_ratio,
            self.masks,
        )
        self.progressive_masks = self.pattern.update_progressive_masks(
            self.pre_masks, self.masks, self.criterion.scores, 1, self.progressive_configs
        )
        self.mask_weights_general(self.progressive_masks)
        if self.progressive_logger:
            self.print_progressive_sparsity()
        return

    def on_step_begin(self, local_step):
        """Update the masks at a given local_step.

        Implement at the start of each step.
        """
        if self.global_step > self.end_step and self.align_masks_flag is False:
            self.align_masks_after_pruning()
            self.align_masks_flag = True

        if self.handled_global_step == self.global_step:
            return

        if not self.use_progressive:
            # As _init_for_progressive() works, when degrades to non-progressive
            # just call BasicPruner's update_masks().
            self.update_masks(local_step)
        else:
            self.update_masks_progressive(local_step)
        self.handled_global_step = self.global_step

    def on_before_optimizer_step(self):
        """Implement before optimizer.step()."""
        self.reg.on_before_optimizer_step()
        self.criterion.on_before_optimizer_step()

    def on_after_optimizer_step(self):
        """Prune the model after optimization."""
        # the order of the following three lines can't not be exchanged
        if self.global_step >= self.start_step and self.global_step <= self.end_step:
            self.reg.on_after_optimizer_step()
        if not self.use_progressive:
            self.mask_weights()
        else:
            self.mask_weights_general(self.progressive_masks)

        self.global_step += 1

    def mask_weights_general(self, input_masks):
        """Apply input masks to corresponding modules' weights.

        Weights are multiplied with input_masks.

        Args:
            input_masks: A dict {"module_name": Tensor} that stores the masks for modules' weights.
        """
        with torch.no_grad():
            for key in self.modules.keys():
                module = self.modules[key]
                module.weight.data = module.weight.data * input_masks[key]

    def print_progressive_sparsity(self):
        """Output the progressive sparsity."""
        cur_sp = self.pattern.get_sparsity_ratio_progressive(self.progressive_masks)
        logger.info("Step: {} -> Current progressive sparsity: {}".format(self.global_step, cur_sp))

    def obtain_weight_sparsity(self, modules):
        total_numels = 0
        sparse_numels = 0
        for key in modules.keys():
            total_numels += modules[key].weight.data.numel()
            sparse_numels += torch.sum(torch.where(modules[key].weight.data == 0, 1, 0)).item()
        return sparse_numels / total_numels

    def align_masks_after_pruning(self):
        if not self.use_progressive:
            return
        """Implement at the end of training phase."""
        # If training ends while a progressive masks is applying, we have to use self.masks to align
        # step 1 calculate sparsity under progressive masks
        sparsity1 = self.obtain_weight_sparsity(self.modules)
        # step 2 use block-wise masks to remask weights
        self.mask_weights_general(self.masks)
        # step 3 calculate sparsity under progressive masks
        sparsity2 = self.obtain_weight_sparsity(self.modules)
        logger.info(f"Replace progressive mask with complete masks: Sparsity Update: {sparsity1} => {sparsity2}")
