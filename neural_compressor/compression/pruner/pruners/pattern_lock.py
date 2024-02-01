"""Pattern lock pruner."""

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

from neural_compressor.utils.logger import Logger

from ..patterns import get_pattern
from .base import KerasBasePruner, PytorchBasePruner, register_pruner

logger = Logger().get_logger()


@register_pruner("pt_pattern_lock")
class PytorchPatternLockPruner(PytorchBasePruner):
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

    def __init__(self, config, modules):
        """Initialize."""
        super().__init__(config, modules)
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
