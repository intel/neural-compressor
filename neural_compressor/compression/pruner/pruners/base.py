"""Base pruner."""

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

import numpy as np

from ..utils import F, safe_get_data, safe_get_grad, safe_get_shape, safe_set_data, tf, torch

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

    def __init__(self, config, modules, framework="pytorch"):
        """Initialize."""
        self.modules = modules
        self.config = config
        self.framework = framework
        self.masks = {}
        self.global_step = 0
        self.handled_global_step = -1
        self.start_step = self.config["start_step"]
        self.end_step = self.config["end_step"]
        self.pruning_frequency = self.config["pruning_frequency"]
        # this is different with original code
        self.total_prune_cnt = (self.end_step - self.start_step + self.pruning_frequency) // self.pruning_frequency
        self.completed_pruned_cnt = 0
        self.total_prune_cnt -= 1  # not pruning at step 0
        if self.total_prune_cnt == 0:
            self.total_prune_cnt = 1
            self.completed_pruned_cnt = 1

        self.target_sparsity_ratio = self.config["target_sparsity"]
        self.current_sparsity_ratio = 0.0
        self.init_sparsity_ratio = 0.0
        self.low_memory_usage = self.config["low_memory_usage"]

    def _init(self):
        """Auxiliary function for initializing."""
        pass

    def on_epoch_begin(self, epoch):
        """Implement at the beginning of each epoch."""
        pass

    def mask_weights(self):
        """Apply masks to corresponding modules' weights.

        Weights are multiplied with masks. This is the formal pruning process.
        """
        pass

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


class PytorchBasePruner(BasePruner):
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

    def __init__(self, config, modules):
        super().__init__(config, modules)
        for key in self.modules.keys():
            module = self.modules[key]
            # TODO: support bias or others
            param_shape = safe_get_shape(module.weight)
            self.masks[key] = torch.ones(param_shape).to(module.weight.device).bool()
        self._init()

    def mask_weights(self):
        """Apply masks to corresponding modules' weights.

        Weights are multiplied with masks. This is the formal pruning process.
        """
        with torch.no_grad():
            for key in self.modules.keys():
                module = self.modules[key]
                param = module.weight
                param_data = safe_get_data(param)
                new_val = param_data * self.masks[key]
                safe_set_data(new_val=new_val, param=param)


class KerasBasePruner(BasePruner):
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

    def __init__(self, config, modules):
        super().__init__(config, modules)
        for key in self.modules.keys():
            module = self.modules[key]
            self.masks[key] = np.ones(module.get_weights()[0].shape)
        self._init()

    def mask_weights(self):
        """Apply masks to corresponding modules' weights.

        Weights are multiplied with masks. This is the formal pruning process.
        """
        for key in self.modules.keys():
            module = self.modules[key]
            module.set_weights([module.get_weights()[0] * self.masks[key]] + module.get_weights()[1:])
