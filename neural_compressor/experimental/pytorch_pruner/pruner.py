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
from .criterias import get_criteria, CRITERIAS
from .regs import get_reg
from .logger import logger


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


def get_pruner(config, modules):
    """Get registered pruner class.
    Get a Pruner object from PRUNERS.
    Args:
        modules: A dict {"module_name": Tensor}. Store the pruning modules' weights.
        config: A config dict object. Contains the pruner information.
    Returns:
        A Pruner object.
    Raises: AssertionError: Cuurently only support pruners which have been registered in PRUNERS.
    """
    
    if "progressive" not in config["prune_type"]:
        name = config["prune_type"]
        config["progressive"] = False
    else:
        # if progressive, delete "progressive" words and reset config["progressive"]
        name = config["prune_type"][0:-12]
        config["progressive"] = True
    if name in CRITERIAS:
        config['criteria_type'] = name
        name = "basic"  ##return the basic pruner

    if name not in PRUNERS.keys():
        assert False, f"does not support {name}, currently only support {PRUNERS.keys()}"
    return PRUNERS[name](config, modules)


class BasePruner:
    """Pruning Pruner.
      The class which executes pruning process.
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
    def __init__(self, config, modules):
        self.modules = modules
        self.config = config
        self.masks = {}
        self.global_step = -1
        self.start_step = self.config['start_step']
        self.end_step = self.config['end_step']
        self.update_frequency_on_step = self.config['update_frequency_on_step']
        ##this is different with original code
        self.total_prune_cnt = (self.end_step - self.start_step + 1) \
                               // self.update_frequency_on_step
        self.completed_pruned_cnt = 0
        for key in self.modules.keys():
            module = self.modules[key]
            self.masks[key] = torch.ones(module.weight.shape).to(module.weight.device)  ##TODO support bias or others

        self.target_sparsity_ratio = self.config['target_sparsity']
        self.current_sparsity_ratio = 0.0
        self.init_sparsity_ratio = 0.0
        self._init()

    def _init(self):
        pass

    def on_epoch_begin(self, epoch):
        pass

    def mask_weights(self):
        with torch.no_grad():
            for key in self.modules.keys():
                module = self.modules[key]
                module.weight.data = module.weight.data * self.masks[key]

    def mask_weights_general(self, input_masks):
        with torch.no_grad():
            for key in self.modules.keys():
                module = self.modules[key]
                module.weight.data = module.weight.data * input_masks[key]

    def on_step_begin(self, local_step):
        pass

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

    def on_before_eval(self):
        pass

    def on_after_eval(self):
        pass

    def check_is_pruned_step(self, step):
        if step < self.start_step or step > self.end_step:
            return False
        if int(step - self.start_step) % self.update_frequency_on_step == 0:
            return True
        return False

    def check_is_pruned_progressive_step(self, step):
        # used in progressive pruning
        if step < self.start_step or step > self.end_step:
            return False
        if int(step - self.start_step) % self.update_frequency_on_step_progressive == 0:
            return True
        return False

@register_pruner("basic")
class BasicPruner(BasePruner):
    """Pruning Pruner.
       The class which executes pruning process.
       1. Defines pruning functions called at step begin/end, epoch begin/end.
       2. Defines the pruning criteria.
       Args:
           modules: A dict {"module_name": Tensor}. Store the pruning modules' weights.
           config: A config dict object. Contains the pruner information.
       Attributes:
           pattern: A Pattern object. Define pruning weights' arrangements within space.
           criteria: A Criteria Object. Define which weights are to be pruned
           scheduler: A Scheduler object. Define model's sparsity changing method as training/pruning executes.
           reg: A Reg object. Define regulization terms.
    """
    def __init__(self, config, modules):
        # self.modules = modules
        # self.config = config
        # self.masks = {}
        super(BasicPruner, self).__init__(config, modules)

    def _init(self):
        self.pattern = get_pattern(self.config, self.modules)
        self.scheduler = get_scheduler(self.config)
        self.criteria = get_criteria(self.config, self.modules)
        self.reg = get_reg(self.config, self.modules, self.pattern)
        # progressive pruning set up, including check up paramters.
        self.use_progressive = self.config["progressive"]
        if self.use_progressive:
            self._init_for_progressive()
        else:
            # if switch off progressive but use per-channel pruning, give a warn
            if "channel" in self.pattern.pattern:
                logger.info("UserWarning: use per-channel pruning pattern without progressive pruning!")
                logger.info("Instead, enabling progressive pruning would be a better choice.")
            else:
                pass

    def _init_for_progressive(self):
        # detailed progressive parameters will stored at patterns.py
        # step 1: check if pattern is NxM
        if "x" not in self.pattern.pattern:
            raise NotImplementedError(f"Currently progressive only support NxM and per-channel pruning patterns.")

        # step 2: check if current set up will "degrade" into non-progressive
        degrading_flag = False
        if (self.end_step - self.start_step) <= self.pattern.progressive_steps or self.pattern.progressive_steps <= 1:
            logger.info("Current progressive setting will degrading to non-progressive pruning.")
            self.use_progressive = False
            return

        # step 3: log hyper-parameters. and check validity.
        if self.use_progressive:
            logger.info(f"Progressive pruning is enabled!")
            logger.info(f"Progressive pruning steps: {self.pattern.progressive_steps}")
            logger.info(f"Progressive type: {self.pattern.progressive_type}")
            logger.info(f"Progressive balance: {self.pattern.use_global}")
            self.pattern.check_progressive_validity()
            self.pre_masks = copy.deepcopy(self.masks)
            self.progressive_masks = copy.deepcopy(self.masks)
            self.update_frequency_on_step_progressive = self.update_frequency_on_step // self.pattern.progressive_steps
            # this is a structural pruning step, it fits self.update_frequency_on_step
            self.structured_update_step = 0

    def set_global_step(self, global_step):
        self.global_step = global_step

    def on_step_begin(self, local_step):
        if not self.use_progressive:
            self.update_masks_nonprogressive(local_step)
        else:
            self.update_masks_progressive(local_step)

    def update_masks_nonprogressive(self, local_step):
        self.global_step += 1

        if self.global_step == self.start_step:
            if self.config['lock_init_sparsity']:
                self.masks = self.pattern.get_pattern_lock_masks(self.modules)
                self.init_sparsity_ratio = self.pattern.get_sparsity_ratio(self.masks)
                self.current_sparsity_ratio = self.init_sparsity_ratio

        if not self.check_is_pruned_step(self.global_step):
            return

        if self.current_sparsity_ratio > self.target_sparsity_ratio:
            return

        self.criteria.on_step_begin()
        current_target_sparsity_ratio = self.scheduler.update_sparsity_ratio(self.target_sparsity_ratio,
                                                                             self.completed_pruned_cnt,
                                                                             self.total_prune_cnt, self.masks,
                                                                             self.init_sparsity_ratio)
        logger.info(f"current target ratio is {current_target_sparsity_ratio}")

        self.completed_pruned_cnt += 1
        if self.criteria.scores == {}:
            return
        self.masks = self.pattern.get_masks(self.criteria.scores, current_target_sparsity_ratio, self.masks,
                                            )
        self.mask_weights()

        self.current_sparsity_ratio = self.pattern.get_sparsity_ratio(self.masks)
        logger.info(f"current sparsity ratio is {self.current_sparsity_ratio}")

    def update_masks_progressive(self, local_step):
        self.global_step += 1

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
        if self.check_is_pruned_progressive_step(self.global_step) and self.check_is_pruned_step(self.global_step) == False:
            # do not do global pruning, only do the progressive mask update.
            step_offset = self.global_step - self.structured_update_step
            progressive_idx = step_offset // self.update_frequency_on_step_progressive
            if progressive_idx < (self.pattern.progressive_steps - 1):
                self.progressive_masks = self.pattern.update_progressive_masks(self.pre_masks, self.masks, self.criteria.scores, progressive_idx+1)
            else:
                # in the end, directly use new masks.
                # logger.info("Use complete mask at the end of pruning process.")
                logger.info(f"{self.global_step} Use complete mask at the end of pruning process.")
                for n in self.masks.keys():
                    self.progressive_masks[n] = self.masks[n].clone()
            current_sparsity = self.pattern.get_sparsity_ratio_progressive(self.progressive_masks)
            # logger.info("Current progressive sparsity is {}".format(current_sparsity))
            logger.info("{} Current progressive sparsity is {}".format(self.global_step, current_sparsity))
            self.mask_weights_general(self.progressive_masks)
            return

        # case 3: a pruning step, generate new masks, progressive masks also update.
        tmp_step = self.global_step
        self.structured_update_step = tmp_step
        current_target_sparsity_ratio = self.scheduler.update_sparsity_ratio(self.target_sparsity_ratio,
                                                                             self.completed_pruned_cnt,
                                                                             self.total_prune_cnt, self.masks)
        logger.info(f"current target ratio is {current_target_sparsity_ratio}")
        self.criteria.on_step_begin()
        self.completed_pruned_cnt += 1
        if self.criteria.scores == {}:
            return
        for n in self.masks.keys():
            self.pre_masks[n] = self.masks[n].clone()
        # update new masks
        self.masks = self.pattern.get_masks(self.criteria.scores, current_target_sparsity_ratio, self.masks,)
        self.progressive_masks = self.pattern.update_progressive_masks(self.pre_masks, self.masks, self.criteria.scores, 1)
        self.mask_weights_general(self.progressive_masks)
        current_sparsity = self.pattern.get_sparsity_ratio_progressive(self.progressive_masks)
        #logger.info("***Current progressive sparsity is {}".format(current_sparsity))
        logger.info("***{} Current progressive sparsity is {}".format(self.global_step, current_sparsity))

    def on_before_optimizer_step(self):
        self.reg.on_before_optimizer_step()

    def on_after_optimizer_step(self):
        ##the order of the following three lines can't not be exchanged
        self.reg.on_after_optimizer_step()
        if not self.use_progressive:
            self.mask_weights()
        else:
            self.mask_weights_general(self.progressive_masks)
        self.criteria.on_after_optimizer_step()


@register_pruner('pattern_lock')
class PatternLockPruner(BasePruner):
    """Pruning Pruner.
    A Pruner class derived from BasePruner. In this pruner, original model's sparsity pattern will be fixed while training.
    This pruner is useful when you want to train a sparse model without change its original structure.
    Args:
        modules: A dict {"module_name": Tensor}. Store the pruning modules' weights.
        config: A config dict object. Contains the pruner information.
    Attributes:
        Inherit from parent class Pruner.
    """
    def __init__(self, config, modules):
        super(PatternLockPruner, self).__init__(config, modules)
        self.pattern = get_pattern(self.config, modules)
        assert self.config.end_step == self.config.start_step, "pattern_lock pruner only supports one shot mode"

    def on_step_begin(self, local_step):
        self.global_step += 1
        if not self.check_is_pruned_step(self.global_step):
            return
        self.masks = self.pattern.get_pattern_lock_masks(self.modules)

    def on_after_optimizer_step(self):
        self.mask_weights()


