"""Pruner."""
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

import re
import copy
import numpy as np
from functools import partial
from .patterns import get_pattern
from .schedulers import get_scheduler
from .criteria import get_criterion, CRITERIA
from .regs import get_reg
from .utils import logger

from ...utils.utility import LazyImport
torch = LazyImport('torch')
tf = LazyImport('tensorflow')
F = LazyImport('torch.nn.functional')

PRUNERS = {}


def register_pruner(name):
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


def parse_valid_pruner_types():
    """Get all valid pruner names."""
    valid_pruner_types = []
    for x in CRITERIA.keys():
        for p in ["", "_progressive"]:
            valid_pruner_types.append(x + p)
    valid_pruner_types.append("pattern_lock")
    return valid_pruner_types

def get_pruner(config, modules, framework='pytorch'):
    """Get registered pruner class.

    Get a Pruner object from PRUNERS.

    Args:
        modules: A dict {"module_name": Tensor} that stores the pruning modules' weights.
        config: A config dict object that contains the pruner information.

    Returns:
        A Pruner object.

    Raises: AssertionError: Cuurently only support pruners that have been registered in PRUNERS.
    """
    ## do the ugly work here
    ## check if it is doing self-multihead-attention pruning
    if "mha" in config["pattern"]:
        return PRUNERS["mha"](config, modules)
    ## if enable progressive pruning or not.
    if "progressive" not in config["pruning_type"]:
        name = config["pruning_type"]
        config["progressive"] = False
    else:
        # if progressive, delete "progressive" words and reset config["progressive"]
        name = config["pruning_type"][0:-12]
        config["progressive"] = True
    if name in CRITERIA:
        if config["progressive"] == False:
            config['criterion_type'] = name
            if "block" in name or "free" in name:
                assert ":" not in config["pattern"], f"{name} pruner type does not support {config['pattern']} pattern."
            else :
                name = "basic"  ##return the basic pruner
        else:
            config['criterion_type'] = name
            # name = "progressive"  ## return the progressive pruner
            name = "progressive"

    if name not in PRUNERS.keys():
        assert False, f"does not support {name}, currently only support {parse_valid_pruner_types()}"
    return PRUNERS[name](config, modules, framework)

class BasePruner:
    """Pruning Pruner.

    The class which executes pruning process.

    Args:
        modules: A dict {"module_name": Tensor} that stores the pruning modules' weights.
        config: A config dict object that contains the pruner information.

    Attributes:
        modules: A dict {"module_name": Tensor} that stores the pruning modules' weights.
        config: A config dict object that contains the pruner information.
        masks: A dict {"module_name": Tensor} that stores the masks for modules' weights.
        scores: A dict {"module_name": Tensor} that stores the score for modules' weights,
            which are used to determine what parts to be pruned by a criterion.
        pattern: A Pattern object defined in ./patterns.py
        scheduler: A scheduler object defined in ./scheduler.py
        current_sparsity_ratio: A float representing the current model's sparsity ratio; it is initialized to be zero.
        global_step: An integer representing the total steps the model has run.
        start_step: An integer representing when to trigger pruning process.
        end_step: An integer representing when to end pruning process.
        pruning_frequency: An integer representing the pruning frequency; it is valid when iterative
            pruning is enabled.
        target_sparsity_ratio: A float showing the final sparsity after pruning.
        max_sparsity_ratio_per_op: A float showing the maximum sparsity ratio for every module.
    """

    def __init__(self, config, modules, framework='pytorch'):
        """Initialize."""
        self.modules = modules
        self.config = config
        self.framework = framework
        self.masks = {}
        self.global_step = 0
        self.handled_global_step = -1
        self.start_step = self.config['start_step']
        self.end_step = self.config['end_step']
        self.pruning_frequency = self.config['pruning_frequency']
        # this is different with original code
        self.total_prune_cnt = (self.end_step - self.start_step + self.pruning_frequency) \
                               // self.pruning_frequency
        self.completed_pruned_cnt = 0
        self.total_prune_cnt -= 1  # not pruning at step 0
        if self.total_prune_cnt == 0:
            self.total_prune_cnt = 1
            self.completed_pruned_cnt = 1

        if self.framework == 'pytorch':
            for key in self.modules.keys():
                module = self.modules[key]
                ##TODO support bias or others
                self.masks[key] = torch.ones(module.weight.shape).to(module.weight.device)
        elif self.framework == 'keras':
            for key in self.modules.keys():
                module = self.modules[key]
                self.masks[key] = np.ones(module.get_weights().shape)
        self.target_sparsity_ratio = self.config['target_sparsity']
        self.current_sparsity_ratio = 0.0
        self.init_sparsity_ratio = 0.0
        self._init()

    def _init(self):
        """Auxiliary function for initializing."""
        pass

    def on_epoch_begin(self, epoch):
        """Implement at the beginning of each epoch."""
        pass

    def mask_weights(self):
        """Apply masks to corresponding modules' weights.

        Weights are multipled with masks. This is the formal pruning process.
        """
        if self.framework == 'pytorch':
            with torch.no_grad():
                for key in self.modules.keys():
                    module = self.modules[key]
                    module.weight.data = module.weight.data * self.masks[key]
        elif self.framework == 'keras':
            for key in self.modules.keys():
                module = self.modules[key]
                module.set_weights(np.array(module.get_weights()) * self.masks[key])

    def mask_weights_general(self, input_masks):
        """Apply input masks to corresponding modules' weights.

        Weights are multipled with input_masks.

        Args:
            input_masks: A dict {"module_name": Tensor} that stores the masks for modules' weights.
        """
        if self.framework == 'pytorch':
            with torch.no_grad():
                for key in self.modules.keys():
                    module = self.modules[key]
                    module.weight.data = module.weight.data * input_masks[key]
        elif self.framework == 'keras':
            for key in self.modules.keys():
                module = self.modules[key]
                module.set_weights(np.array(module.get_weights()) * input_masks[key])

    def on_step_begin(self, local_step):
        """Implement at the start of each step."""
        if self.handled_global_step == self.global_step:
            return
        self.update_masks(local_step)
        self.handled_global_step = self.global_step

    def update_masks(self, local_step):
        """Update the masks at a given local step."""
        pass

    def on_epoch_end(self):
        """Implement at the end of each epoch."""
        pass

    def on_step_end(self):
        """Implement at the end of each step."""
        pass

    def on_before_optimizer_step(self):
        """Implement before optimizer.step()."""
        pass

    def on_after_optimizer_step(self):
        """Implement after optimizer.step().

        Prune the model after optimization.
        """
        self.mask_weights()
        self.global_step += 1

    def on_train_begin(self, dataloader=None):
        """Implement at the beginning of training phase."""
        pass

    def on_train_end(self):
        """Implement at the end of training phase."""
        pass

    def on_before_eval(self):
        """Implement at the beginning of evaluation phase."""
        pass

    def on_after_eval(self):
        """Implement at the end of evaluation phase."""
        pass

    def check_is_pruned_step(self, step):
        """Check if a pruning process should be performed at the current step.

        Args:
            step: an integer representing the number of current step.

        Returns:
            A Boolean.
        """
        if step < self.start_step or step > self.end_step:
            return False
        if int(step - self.start_step) % self.pruning_frequency == 0:
            return True
        return False

    def rewrite_forward(self):
        """Rewrite forward to implement block mask operation"""
        assert self.framework != 'keras', "This pruning method is not supported by Keras now."
        def forward(self, input):
            block_size = [self.weight.shape[0]//self.block_mask.shape[0], \
                    self.weight.shape[1]//self.block_mask.shape[1]]
            mask = self.block_mask.repeat_interleave(block_size[0], dim=0).repeat_interleave(\
                                                        block_size[1], dim=-1).to(self.weight.device)
            return F.linear(input, self.weight*mask, self.bias)

        for key in self.modules.keys():
                if not hasattr(self.modules[key], 'block_mask'):
                    continue # No corresponding block mask, skip.
                module = self.modules[key]
                module.forward = partial(forward, module)

    def recover_forward(self):
        """Restore the forward format at the end of pruning"""
        assert self.framework != 'keras', "This pruning method is not supported by Keras now."
        with torch.no_grad():
            for key in self.modules.keys():
                if not hasattr(self.modules[key], 'block_mask'):
                    continue # No corresponding block mask, skip.
                module = self.modules[key]
                module.forward = partial(torch.nn.Linear.forward, module)


@register_pruner("basic")
class BasicPruner(BasePruner):
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

    def __init__(self, config, modules, framework='pytorch'):
        """Initialize."""
        # self.modules = modules
        # self.config = config
        # self.masks = {}
        super(BasicPruner, self).__init__(config, modules, framework)

    def _init(self):
        """Auxiliary function for initializing."""
        self.pattern = get_pattern(self.config, self.modules, self.framework)
        self.scheduler = get_scheduler(self.config)
        self.criterion = get_criterion(self.config, self.modules, self.framework)
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
            if self.config['lock_init_sparsity']:
                self.masks = self.pattern.get_pattern_lock_masks(self.modules)
                self.init_sparsity_ratio = self.pattern.get_sparsity_ratio(self.masks)
                self.current_sparsity_ratio = self.init_sparsity_ratio

        if not self.check_is_pruned_step(self.global_step):
            return

        if self.current_sparsity_ratio > self.target_sparsity_ratio:
            return

        self.criterion.on_step_begin()
        current_target_sparsity_ratio = self.scheduler.update_sparsity_ratio(self.target_sparsity_ratio,
                                                                             self.completed_pruned_cnt,
                                                                             self.total_prune_cnt, self.masks,
                                                                             self.init_sparsity_ratio)
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


@register_pruner('pattern_lock')
class PatternLockPruner(BasePruner):
    """Pruning Pruner.

    A Pruner class derived from BasePruner.
    In this pruner, original model's sparsity pattern will be fixed while training.
    This pruner is useful when a user trains a sparse model without changing its original structure.

    Args:
        modules: A dict {"module_name": Tensor} that stores the pruning modules' weights.
        config: A config dict object that contains the pruner information.

    Attributes:
        Inherit from parent class Pruner.
    """

    def __init__(self, config, modules, framework='pytorch'):
        """Initialize."""
        super(PatternLockPruner, self).__init__(config, modules, framework)
        self.pattern = get_pattern(self.config, modules)
        assert self.config.end_step == self.config.start_step, "pattern_lock pruner only supports one shot mode"

    def update_masks(self, local_step):
        """Update the masks at a given local step."""
        if not self.check_is_pruned_step(self.global_step):
            return
        self.masks = self.pattern.get_pattern_lock_masks(self.modules)

    def on_after_optimizer_step(self):
        """Implement after optimizer.step().

        Prune the model after optimization.
        """
        self.mask_weights()
        self.global_step += 1


@register_pruner('block_mask')
class BlockMaskPruner(BasePruner):
    """Pruning Pruner.

    The class which executes pruning process.
    1. Defines pruning functions called at step begin/end, before/after optimize and epoch begin/end.
    2. Defines the pruning criterion.
    3. Obtain block masks and its grads.

    Args:
        modules: A dict {"module_name": Tensor} that stores the pruning modules' weights.
        config: A config dict object that contains the pruner information.

    Attributes:
        pattern: A Pattern object that defines pruning weights' arrangements within space.
        criterion: A Criterion Object that defines which weights are to be pruned
        scheduler: A Scheduler object that defines how the model's sparsity changes as training/pruning proceeds.
        reg: A Reg object that defines regulization terms.
    """
    def __init__(self, config, modules, framework='pytorch'):
        """Initialize."""
        super(BlockMaskPruner, self).__init__(config, modules, framework)

    def _init(self):
        """Initialize."""
        self.pattern = get_pattern(self.config, self.modules, self.framework)
        self.masks = self.pattern.register_block_masks(self.modules)
        self.rewrite_forward()
        self.scheduler = get_scheduler(self.config)
        self.criterion = get_criterion(self.config, self.modules, self.framework)
        self.reg = get_reg(self.config, self.modules, self.pattern)

        if "channel" not in self.pattern.pattern:
            logger.info("Enabling channel-wise pattern would be a better choice.")

    # def on_step_begin(self, local_step):
    #     """Implement at the start of each step.

    #     Update the masks at a given local_step.
    #     """
    #     self.update_masks(local_step)

    def update_masks(self, local_step):
        """Update the masks at a given local step."""
        if self.global_step == self.start_step:
            if self.config['lock_init_sparsity']:
                self.init_sparsity_ratio = self.pattern.get_sparsity_ratio(self.masks)
                self.current_sparsity_ratio = self.init_sparsity_ratio

        if not self.check_is_pruned_step(self.global_step):
            return

        if self.current_sparsity_ratio > self.target_sparsity_ratio:
            return

        self.criterion.on_step_begin()
        current_target_sparsity_ratio = self.scheduler.update_sparsity_ratio(self.target_sparsity_ratio,
                                                                             self.completed_pruned_cnt,
                                                                             self.total_prune_cnt, self.masks,
                                                                             self.init_sparsity_ratio)
        logger.info(f"current target ratio is {current_target_sparsity_ratio}")

        self.completed_pruned_cnt += 1
        if self.criterion.scores == {}:
            return
        self.masks = self.pattern.get_masks(self.criterion.scores, current_target_sparsity_ratio, self.masks)
        self.update_block_masks(self.masks)
        self.mask_weights()

        self.current_sparsity_ratio = self.pattern.get_sparsity_ratio(self.masks)
        logger.info(f"current sparsity ratio is {self.current_sparsity_ratio}")

    def on_before_optimizer_step(self):
        """Implement before optimizer.step()."""
        if self.global_step >= self.start_step and self.global_step <= self.end_step:
            self.reg.on_before_optimizer_step()
            self.criterion.on_before_optimizer_step()

    def on_after_optimizer_step(self):
        """Prune the model after optimization."""
        ##the order of the following four lines can't not be exchanged
        if self.global_step >= self.start_step and self.global_step <= self.end_step:
            self.reg.on_after_optimizer_step()
        self.zero_mask_grad()
        self.mask_weights()
        if not self.end_step or self.end_step == self.global_step:
            # recover forward method and remove block mask parameters at last prune step
            self.recover_forward()
            self.pattern.remove_block_masks()
        self.global_step += 1

    def mask_weights(self):
        """Apply block masks to corresponding modules' weights.

        Weights are multipled with masks. This is the formal pruning process.
        """
        with torch.no_grad():
            self.pattern.mask_block_weights(self.masks)

    def update_block_masks(self, masks):
        """Update the block mask parameters."""
        with torch.no_grad():
            for key in self.masks.keys():
                module = self.modules[key]
                module.block_mask.data = masks[key].data

    def zero_mask_grad(self):
        with torch.no_grad():
            for key in self.modules.keys():
                if not hasattr(self.modules[key], 'block_mask'):
                    continue # No corresponding block mask, skip.
                mask = self.modules[key].block_mask
                if mask.grad is not None:
                    if mask.grad.grad_fn is not None:
                        mask.grad.detach_()
                    else:
                        mask.grad.requires_grad_(False)
                    mask.grad.zero_()


@register_pruner('retrain_free')
class RetrainFreePruner(BasePruner):
    """Pruning Pruner.
    The retrain_free pruner_class is derived from BasePruner.
    This pruner references the mask search and mask rearrangement strategies in fast retraining free.
    RetrainFreePruner supports one-shot pruning (same effect as fast retraining free) and iterative pruning.
    Please refer to A Fast Post-Training Pruning Framework for Transformers
        (https://arxiv.org/abs/2204.09656)

    1. Defines pruning functions called at step begin/end, before/after optimize and epoch begin/end.
    2. Defines the pruning criterion and fixed weight parameters.
    3. Obtain block masks and its grads.
    4. Rearrange block masks.

    Args:
        modules: A dict {"module_name": Tensor} that stores the pruning modules' weights.
        config: A config dict object that contains the pruner information.

    Attributes:
        pattern: A Pattern object that defines pruning weights' arrangements within space.
        criterion: A Criterion Object that defines which weights are to be pruned
        scheduler: A Scheduler object that defines how the model's sparsity changes as training/pruning proceeds.
        reg: A Reg object that defines regulization terms.
    """
    def __init__(self, config, modules, framework='pytorch'):
        """Initialize."""
        super(RetrainFreePruner, self).__init__(config, modules, framework)

    def _init(self):
        """Initialize."""
        self.pattern = get_pattern(self.config, self.modules, self.framework)
        self.masks = self.pattern.register_block_masks(self.modules)
        self.rewrite_forward()
        self.scheduler = get_scheduler(self.config)
        self.criterion = get_criterion(self.config, self.modules, self.framework)
        self.reg = get_reg(self.config, self.modules, self.pattern)

        logger.warning("Retrain-free pruner fixed the weights, please DO NOT turn on gradient update.")
        assert "channel" in self.pattern.pattern, \
                "retrain-free pruner only supports large patterns like channel-wise pruning."

    # def on_step_begin(self, local_step):
    #     """Implement at the start of each step.

    #     Update the masks at a given local_step.
    #     """
    #     self.update_masks(local_step)

    def update_masks(self, local_step):
        """Update the masks at a given local step."""
        if self.global_step == self.start_step:
            if self.config['lock_init_sparsity']:
                self.init_sparsity_ratio = self.pattern.get_sparsity_ratio(self.masks)
                self.current_sparsity_ratio = self.init_sparsity_ratio

        if not self.check_is_pruned_step(self.global_step):
            return

        if self.current_sparsity_ratio > self.target_sparsity_ratio:
            return

        self.criterion.on_step_begin()
        current_target_sparsity_ratio = self.scheduler.update_sparsity_ratio(self.target_sparsity_ratio,
                                                                             self.completed_pruned_cnt,
                                                                             self.total_prune_cnt, self.masks,
                                                                             self.init_sparsity_ratio)
        logger.info(f"current target ratio is {current_target_sparsity_ratio}")

        self.completed_pruned_cnt += 1
        if self.criterion.scores == {}:
            return
        # the order of the following three lines can't not be exchanged
        self.masks = self.pattern.get_masks(self.criterion.scores, current_target_sparsity_ratio, self.masks)
        self.rearrange_masks(self.masks)
        self.update_block_masks(self.masks)

        self.current_sparsity_ratio = self.pattern.get_sparsity_ratio(self.masks)
        logger.info(f"current sparsity ratio is {self.current_sparsity_ratio}")

    def on_before_optimizer_step(self):
        """Implement before optimizer.step()."""
        if self.global_step >= self.start_step and self.global_step <= self.end_step:
            self.reg.on_before_optimizer_step()
            self.criterion.on_before_optimizer_step()

    def on_after_optimizer_step(self):
        """Prune the model after optimization."""
        ##the order of the following four lines can't not be exchanged
        if self.global_step >= self.start_step and self.global_step <= self.end_step:
            self.reg.on_after_optimizer_step()
        # self.mask_weights() 
        # Iterative rearrangement with mask weight at the last step only
        if self.end_step == self.global_step:
            self.mask_weights()
            logger.info(f"mask weights at last_prune_step: {self.global_step}")
            # recover forward method and remove block mask parameters at last prune step
            self.recover_forward()
            self.pattern.remove_block_masks()
        self.global_step += 1

    def mask_weights(self):
        """Apply block masks to corresponding modules' weights.

        Weights are multipled with masks. This is the formal pruning process.
        """
        with torch.no_grad():
            self.pattern.mask_block_weights(self.masks)

    def update_block_masks(self, masks):
        """Update the block mask parameters."""
        with torch.no_grad():
            for key in self.masks.keys():
                module = self.modules[key]
                module.block_mask.data = masks[key].data

    def rearrange_masks(self, masks):
        """Rearrange the masks of each layer with constant sparsity."""
        with torch.no_grad():
            new_masks = {}
            for key in masks.keys():
                block_mask = masks[key]
                num_pruned = torch.sum(block_mask == 0.0).data.item()
                if not num_pruned or not self.criterion.collected_grads[key]:
                    new_masks[key] = block_mask
                    continue
                grads = torch.stack(self.criterion.collected_grads[key], dim=0).squeeze()
                grads = grads.permute(1, 0).contiguous()
                grads_sq = grads.pow(2).sum(dim=1)
                _, indicies = grads_sq.sort(descending=False)
                indicies = indicies.tolist()
                masked_indicies = indicies[:num_pruned]
                for index in indicies[num_pruned:]:
                    masked_indicies.append(index)
                    grad_vectors = grads[masked_indicies]
                    grad_sum = grad_vectors.sum(dim=0)
                    complement = grad_sum - grad_vectors
                    grad_sum_length = complement.pow(2).sum(dim=1)
                    removed = grad_sum_length.argmin()
                    del masked_indicies[removed]

                new_masks[key] = torch.ones(len(indicies)).to(block_mask.device)
                new_masks[key][masked_indicies] = 0
                new_masks[key] = new_masks[key] * torch.ones_like(block_mask).to(block_mask.device)
            self.masks = new_masks

    def zero_mask_grad(self):
        with torch.no_grad():
            for key in self.modules.keys():
                if not hasattr(self.modules[key], 'block_mask'):
                    continue # No corresponding block mask, skip.
                mask = self.modules[key].block_mask
                if mask.grad is not None:
                    if mask.grad.grad_fn is not None:
                        mask.grad.detach_()
                    else:
                        mask.grad.requires_grad_(False)
                    mask.grad.zero_()

@register_pruner('progressive')
class ProgressivePruner(BasicPruner):
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

    def __init__(self, config, modules, framework='pytorch'):
        """Initialize."""
        super(ProgressivePruner, self).__init__(config, modules, framework)

    def _init(self):
        """Auxiliary function for initialization."""
        self.pattern = get_pattern(self.config, self.modules, self.framework)
        self.scheduler = get_scheduler(self.config)
        self.criterion = get_criterion(self.config, self.modules, self.framework)
        self.reg = get_reg(self.config, self.modules, self.pattern)
        # progressive pruning set up, including check up paramters.
        self.use_progressive = self.config["progressive"]
        # progressive parameters
        # dict passed to Pattern's functions
        self.progressive_configs = {
            "progressive_steps": 4,
            "progressive_type": "scores",
            "use_global": True
        }
        self.progressive_steps = self.progressive_configs["progressive_steps"]
        self.progressive_type = self.progressive_configs["progressive_type"]
        self.use_global = self.progressive_configs["use_global"]
        self.progressive_logger = False
        self._init_for_progressive()

    def _init_for_progressive(self):
        """Auxiliary function for initializing progressive pruning."""
        # detailed progressive parameters will stored at patterns.py
        # step 1: check if pattern is NxM
        # if "x" not in self.pattern.pattern:
        #     raise NotImplementedError(f"Currently progressive only " \
        #                               f"support NxM and per-channel pruning patterns.")

        # step 2: check if current set up will "degrade" into non-progressive
        degrading_flag = False
        if (self.end_step - self.start_step) <= self.progressive_steps or self.progressive_steps <= 1:
            logger.info("Current progressive setting will degrading to non-progressive pruning.")
            self.use_progressive = False
            return

        # step 3: log hyper-parameters. and check validity.
        if self.use_progressive:
            logger.info(f"Progressive pruning is enabled!")
            logger.info(f"Progressive pruning steps: {self.progressive_steps}")
            logger.info(f"Progressive type: {self.progressive_type}")
            logger.info(f"Progressive balance: {self.use_global}")
            self.check_progressive_validity()
            self.pre_masks = copy.deepcopy(self.masks)
            self.progressive_masks = copy.deepcopy(self.masks)
            if self.pruning_frequency < self.progressive_steps:  ##TODO trick
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
            assert type(self.pattern).__name__ == "PatternNxM", "Progressive linear pruning only support NxM."
            if self.use_global:
                # when global progressive is applied, linear type is contradict.
                raise NotImplementedError("Global progressive pruning do not support linear pattern")
            # When linear, progressive_step should not meet a indivisible
            for key in self.pattern.block_size.keys():
                block_size = self.pattern.block_size[key]
                progressive_direction = max(block_size)
                if progressive_direction % self.progressive_steps != 0:
                    raise ValueError(
                        f"In layer {key}, its pruning pattern is {block_size}, " \
                        f"while progressive steps {self.progressive_steps} is indivisible.")
        else:
            # score based progressive pruning, support both NxM and N:M patterns
            if type(self.pattern).__name__ == "PatternNxM":
                for key in self.pattern.block_size.keys():
                    block_size = self.pattern.block_size[key]
                    total_block_size = block_size[0] * block_size[1]
                    if total_block_size < self.progressive_steps:
                        raise ValueError(
                            f"In layer {key}, its pruning pattern is {block_size}, " \
                            f"while progressive steps {self.progressive_steps} is overflowing.")
            elif type(self.pattern).__name__ == "PatternNInM":
                if self.pattern.N < self.progressive_steps:
                    raise ValueError(
                            f"Pruning pattern is {self.pattern.N} in {self.pattern.M}, " \
                            f"while progressive steps {self.progressive_steps} is overflowing.")
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
            if self.config['lock_init_sparsity']:
                self.masks = self.pattern.get_pattern_lock_masks(self.modules)
                self.init_sparsity_ratio = self.pattern.get_sparsity_ratio(self.masks)
                self.current_sparsity_ratio = self.init_sparsity_ratio

        # case 1: step is not in [start_step, end_step] or it is not either pruning or progressive pruning step.
        if (self.check_is_pruned_step(self.global_step) == False) and (
                self.check_is_pruned_progressive_step(self.global_step) == False):
            return
        if self.current_sparsity_ratio > self.target_sparsity_ratio:
            return

        # case 2: step which does progressive update, but it is not a pruning step in case 3
        if self.check_is_pruned_progressive_step(self.global_step) \
                and self.check_is_pruned_step(self.global_step) == False:
            # do not do global pruning, only do the progressive mask update.
            step_offset = self.global_step - self.structured_update_step
            progressive_idx = step_offset // self.pruning_frequency_progressive
            if progressive_idx < (self.progressive_steps - 1):
                self.progressive_masks = self.pattern.update_progressive_masks(self.pre_masks, self.masks, \
                                                                               self.criterion.scores, \
                                                                               progressive_idx + 1, \
                                                                               self.progressive_configs)
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
        current_target_sparsity_ratio = self.scheduler.update_sparsity_ratio(self.target_sparsity_ratio,
                                                                             self.completed_pruned_cnt,
                                                                             self.total_prune_cnt, self.masks)
        logger.info(f"current target ratio is {current_target_sparsity_ratio}")
        self.criterion.on_step_begin()
        self.completed_pruned_cnt += 1
        if self.criterion.scores == {}:
            return
        for n in self.masks.keys():
            self.pre_masks[n] = self.masks[n].clone()
        # update new masks
        self.masks = self.pattern.get_masks(self.criterion.scores, current_target_sparsity_ratio, self.masks, )
        self.progressive_masks = self.pattern.update_progressive_masks(self.pre_masks, self.masks, \
                                                                       self.criterion.scores, 1, \
                                                                       self.progressive_configs)
        self.mask_weights_general(self.progressive_masks)
        if self.progressive_logger:
            self.print_progressive_sparsity()
        return

    def on_step_begin(self, local_step):
        """Update the masks at a given local_step.

        Implement at the start of each step.
        """
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
        ##the order of the following three lines can't not be exchanged
        if self.global_step >= self.start_step and self.global_step <= self.end_step:
            self.reg.on_after_optimizer_step()
        if not self.use_progressive:
            self.mask_weights()
        else:
            self.mask_weights_general(self.progressive_masks)

        self.global_step += 1

    def print_progressive_sparsity(self):
        """Output the progressive sparsity."""
        cur_sp = self.pattern.get_sparsity_ratio_progressive(self.progressive_masks)
        logger.info("Step: {} -> Current progressive sparsity: {}".format(self.global_step, cur_sp))

@register_pruner('mha')
class MultiheadAttentionPruner(BasePruner):
    """Pruning Pruner.

    In this pruner, We apply pruning for multi-head attentions.
    multi-head attention pruning means remove partial QKV layers and their corresponding feedward layers simultaneously. 

    Args:
        mha_modules: A List 
        [
            {
                'qkv_name': ['query_layer_name', 'key_layer_name', 'value_layer_name'],
                'ffn_name': ['attention_ffn_name'],
                'mha_name': ['mha_name'] (keep not change),
                'qkv_module': [torch.nn.Linear, torch.nn.Linear, torch.nn.Linear],
                'ffn_module': [torch.nn.Linear],
                'mha_module': [torch.nn.Module] (keep not change),
            }
            ...
        ]
        that stores the pruning mha modules.
        config: A config dict object that contains the pruner information.

    Attributes:
        Inherit from parent class Pruner.
    """
    def __init__(self, config, mha_modules):
        """Initialize."""
        # use pattern search techique to obtain multihead attention modules
        # modules is a dict that fits the mha auto slim process
        #----------------------------------------- 
        self.config = config
        self.mha_modules = mha_modules
        self.global_step = 0
        self.handled_global_step = -1
        self.start_step = self.config['start_step']
        self.end_step = self.config['end_step']
        self.pruning_frequency = self.config['pruning_frequency']
        ##this is different with original code
        self.total_prune_cnt = (self.end_step - self.start_step + self.pruning_frequency) \
                               // self.pruning_frequency
        self.completed_pruned_cnt = 0
        self.total_prune_cnt -= 1  ## not pruning at step 0
        if self.total_prune_cnt == 0:
            self.total_prune_cnt = 1
            self.completed_pruned_cnt = 1
        self.target_sparsity_ratio = self.config['target_sparsity']
        self.current_sparsity_ratio = 0.0
        self.init_sparsity_ratio = 0.0
        self.criterion_reduce_type = self.config['criterion_reduce_type']
        self.pruning_scope = self.config['pruning_scope']
        #-----------------------------------------------------------------------------------------------
        #---------------------------------------Custom attributes for MHA Pruner
        # main initialize process.
        # define some attributes. 
        # {key: mha_name, value: mha_compression object}
        self.mha_compressions = {}
        # {key: layer_name, value: corresponding linear object}
        self.linear_layers = {}
        # {key: mha_name, value: torch.Tensor, 1xhead_num}, head_num traced in corresponding mha_compression object
        self.head_masks = {} 
        # general pruning components (head pruning does not need a pattern component)
        self.mha_scores = {} # {}
        # main initialization process
        # initialize custom attributes
        self._init_mha_attrs()
        # initialize custom attributes: criterion (snip-momnetum, snip, magnitude, etc.)
        # we have hook modules in mha_compressions therefore do not pass them to patterns
        self.pattern = get_pattern(self.config, modules = None) 
        self.criterion = get_criterion(self.config, self.linear_layers) # criterion hooks on linear themselves.
        self.scheduler = get_scheduler(self.config)
        #-----------------------------------------------------------------------------------------------
    
    def _init_mha_attrs(self):
        # initialize self.mha_compressions, self.linear_layers, self.head_masks
        # similar to original mha slim process, but only hook mha modules and their attributes, 
        # do not call slim main functions.
        
        # auto slim related: head pruning objects
        from .model_slim.weight_slim import MHACompression
        for mha_module in self.mha_modules:
            # initialize self.mha_compressions
            mha_comp = MHACompression(mha_module)
            self.mha_compressions[mha_module['mha_name'][0]] = mha_comp
            head_nums_for_this_mha =  getattr(mha_comp.mha[0], mha_comp.attributes_for_this_mha['head_nums'])
            # initialize head_masks
            # why use 1 x head_num shape? because this provides convenience for permute mask for qkv and ffn
            self.head_masks[mha_module['mha_name'][0]] = torch.ones(1, head_nums_for_this_mha)
            # initialize self.linear_layers
            for idx in range(mha_module['qkv_name'].__len__()):
                # update qkv layers
                self.linear_layers[mha_module['qkv_name'][idx]] = mha_module['qkv_module'][idx]
            for idx in range(mha_module['ffn_name'].__len__()):
                self.linear_layers[mha_module['ffn_name'][idx]] = mha_module['ffn_module'][idx]
    
    def reduce_mha_scores(self, score, dim = 0):
        # an 2D tensor, return its compiled scores
        if self.criterion_reduce_type == "mean":
            return torch.mean(score, dim)
        elif self.criterion_reduce_type == "sum":
            return torch.sum(score, dim)
        elif self.criterion_reduce_type == "max":
            return torch.max(score, dim)
        else:
            raise NotImplementedError
    
    def print_mha_masks(self):
        for k, v in self.head_masks.items():
            logger.info(f"Head mask of module {k} is {v}.")

    def update_mha_scores(self):
        for mha_name, mha_comp in self.mha_compressions.items():
            device = mha_comp.device
            # step 0: obtain hooked attributes in mha modules
            head_size = getattr(mha_comp.mha[0], mha_comp.attributes_for_this_mha['head_size'])
            head_nums = getattr(mha_comp.mha[0], mha_comp.attributes_for_this_mha['head_nums'])
            # step 1: gather qkv and ffn which belong to same mha together
            qkv_scores_for_this_mha = {}
            ffn_scores_for_this_mha = {}
            for layer_name, layer_score in self.criterion.scores.items():
                if layer_name in mha_comp.qkv_name:
                    qkv_scores_for_this_mha[layer_name] = layer_score
                elif layer_name in mha_comp.ffn_name:
                    ffn_scores_for_this_mha[layer_name] = layer_score
                else:
                    continue
            # step 2: get qkv and ffn reduce_dim scores (commonly use: mean)
            qkv_gather_scores = torch.zeros(head_nums, 1).to(device)
            qkv_shape = mha_comp.qkv[0].weight.shape
            qkv_block_size = [head_size, qkv_shape[1]]
            qkv_new_shape = [
                qkv_shape[0] // qkv_block_size[0], 
                qkv_block_size[0], 
                qkv_shape[1] // qkv_block_size[1], 
                qkv_block_size[1]
            ]
            for qkv_name, qkv_score in qkv_scores_for_this_mha.items():
                qkv_score_new = qkv_score.reshape(qkv_new_shape)
                qkv_score_new = self.reduce_mha_scores(self.reduce_mha_scores(qkv_score_new, -1), 1)
                # qkv_scores_for_this_mha[qkv_name] = qkv_score_new # [head_nums, 1]
                qkv_gather_scores += qkv_score_new
            ffn_gather_scores = torch.zeros(1, head_nums).to(device)
            ffn_shape = mha_comp.ffn[0].weight.shape
            ffn_block_size = [ffn_shape[0], head_size]
            ffn_new_shape = [
                ffn_shape[0] // ffn_block_size[0], 
                ffn_block_size[0], 
                ffn_shape[1] // ffn_block_size[1], 
                ffn_block_size[1]
            ]
            for ffn_name, ffn_score in ffn_scores_for_this_mha.items():
                ffn_score_new = ffn_score.reshape(ffn_new_shape)
                ffn_score_new = self.reduce_mha_scores(self.reduce_mha_scores(ffn_score_new, -1), 1)
                # ffn_scores_for_this_mha[ffn_name] = ffn_score_new # [1, head_nums]
                ffn_gather_scores += ffn_score_new
            # step 3: compile qkv ffn scores to obtain individual head's score
            self.mha_scores[mha_name] = qkv_gather_scores + ffn_gather_scores.permute(1, 0)
            self.mha_scores[mha_name] /= (len(qkv_scores_for_this_mha) + len(ffn_scores_for_this_mha)) # should be 4
        return True

    def update_masks(self, local_step):
        """Update the masks at a given local step."""
        if self.global_step == self.start_step:
            if self.config['lock_init_sparsity']:
                self.masks = self.pattern.get_pattern_lock_masks(self.modules)
                self.init_sparsity_ratio = self.pattern.get_sparsity_ratio(self.masks)
                self.current_sparsity_ratio = self.init_sparsity_ratio

        if not self.check_is_pruned_step(self.global_step):
            return

        if self.current_sparsity_ratio > self.target_sparsity_ratio:
            return

        self.criterion.on_step_begin()
        current_target_sparsity_ratio = self.scheduler.update_sparsity_ratio(self.target_sparsity_ratio,
                                                                             self.completed_pruned_cnt,
                                                                             self.total_prune_cnt, 
                                                                             self.head_masks,
                                                                             self.init_sparsity_ratio)
        logger.info(f"current target ratio is {current_target_sparsity_ratio}")

        self.completed_pruned_cnt += 1
        if self.criterion.scores == {}:
            return

        # self.masks = self.pattern.get_masks(self.criterion.scores, current_target_sparsity_ratio, self.masks)
        # self.mask_weights()
        self.update_mha_scores() # update self.mha_scores
        self.head_masks = self.pattern.get_masks(self.mha_scores, current_target_sparsity_ratio, self.head_masks)
        self.print_mha_masks()
        self.mask_weights()

        self.current_sparsity_ratio = self.pattern.get_sparsity_ratio(self.head_masks)
        logger.info(f"current sparsity ratio is {self.current_sparsity_ratio}")
    
    def mask_weights(self):
        for mha_name, mha_compression in self.mha_compressions.items():
            mha_compression.mask_mha_weights(self.head_masks[mha_name])

    # main api functions
    def on_step_begin(self, local_step):
        """Implement at the start of each step."""
        if self.handled_global_step == self.global_step:
            return
        self.update_masks(local_step)
        self.handled_global_step = self.global_step

    def on_before_optimizer_step(self):
        """Implement before optimizer.step()."""
        self.criterion.on_before_optimizer_step()

    def on_after_optimizer_step(self):
        """Implement after optimizer.step().

        Prune the model after optimization.
        """
        self.mask_weights()
        self.global_step += 1
