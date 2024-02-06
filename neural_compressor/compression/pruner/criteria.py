"""Pruning criterion."""

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

from .utils import safe_get_data, safe_get_grad, safe_get_shape, torch

CRITERIA = {}


def register_criterion(name):
    """Register a criterion to the registry."""

    def register(criterion):
        CRITERIA[name] = criterion
        return criterion

    return register


def get_criterion(config, modules, pattern, masks=None):
    """Get registered criterion class."""
    name = config["criterion_type"]
    if name not in CRITERIA.keys():
        assert False, f"criteria does not support {name}, currently only support {CRITERIA.keys()}"
    if masks is not None:
        return CRITERIA[name](modules, config, pattern, masks)
    return CRITERIA[name](modules, config, pattern)


class PruningCriterion:
    """Pruning base criterion.

    Args:
        config: A config dict object that includes information about pruner and pruning criterion.
        modules: A dict {"module_name": Tensor} that stores the pruning modules' weights.

    Attributes:
        scores: A dict {"module_name": Tensor} that stores the scores of pruning modules.
    """

    def __init__(self, modules, config, pattern):
        """Initialize a pruning criterion."""
        self.scores = {}
        self.modules = modules
        self.config = config
        self.pattern = pattern
        self.low_memory_usage = config["low_memory_usage"]

    def on_step_begin(self):
        """Calculate and store the pruning scores of pruning modules at the beginning of a step."""
        pass

    def on_before_optimizer_step(self):
        """Calculate and store the pruning scores of pruning modules before the optimizer step."""
        pass

    def on_after_optimizer_step(self):
        """Calculate and store the pruning scores of pruning modules after the optimizer step."""
        pass


@register_criterion("magnitude")
class MagnitudeCriterion(PruningCriterion):
    """Pruning criterion.

    The magnitude criterion_class is derived from PruningCriterion.
    The magnitude value is used to score and determine if a weight is to be pruned.

    Args:
        config: A config dict object that includes information about pruner and pruning criterion.
        modules: A dict {"module_name": Tensor} that stores the pruning modules' weights.

    Attributes:
        scores: A dict {"module_name": Tensor} that stores the scores of pruning modules.
    """

    def __init__(self, modules, config, pattern):
        """Initialize a magnitude pruning criterion."""
        super(MagnitudeCriterion, self).__init__(modules, config, pattern)

    def on_step_begin(self):
        """Calculate and store the pruning scores based on a magnitude criterion."""
        with torch.no_grad():
            for key in self.modules.keys():
                param = self.modules[key].weight
                p = safe_get_data(param)
                if hasattr(self.pattern, "reduce_score"):
                    self.scores[key] = self.pattern.reduce_score(torch.abs(p), key)
                else:
                    self.scores[key] = torch.abs(p)


@register_criterion("gradient")
class GradientCriterion(PruningCriterion):
    """Pruning criterion.

    The gradient criterion_class is derived from PruningCriterion.
    The absolute value of gradient is used to score and determine if a weight is to be pruned.

    Args:
        config: A config dict object that includes information about pruner and pruning criterion.
        modules: A dict {"module_name": Tensor} that stores the pruning modules' weights.

    Attributes:
        scores: A dict {"module_name": Tensor} that stores the scores of pruning modules.
    """

    def __init__(self, modules, config, pattern):
        """Initialize a gradient pruning criterion."""
        super(GradientCriterion, self).__init__(modules, config, pattern)
        assert self.config.end_step > 0, "please set end_step > 0 for gradient based criterion"

    def on_before_optimizer_step(self):
        """Calculate and store the pruning scores based on gradient criterion."""
        with torch.no_grad():
            for key in self.modules.keys():
                p = self.modules[key].weight
                # self.scores[key] = torch.abs(p.grad)
                if hasattr(self.pattern, "reduce_score"):
                    self.scores[key] = self.pattern.reduce_score(torch.abs(p.grad), key)
                else:
                    self.scores[key] = torch.abs(p.grad)


@register_criterion("snip")
class SnipCriterion(PruningCriterion):
    """Pruning criterion.

    The snip criterion_class is derived from PruningCriterion.
    The product of magnitude and gradient is used to score and determine if a weight is to be pruned.
    Please refer to SNIP: Single-shot Network Pruning based on Connection Sensitivity.
    (https://arxiv.org/abs/1810.02340)

    Args:
        config: A config dict object that includes information about pruner and pruning criterion.
        modules: A dict {"module_name": Tensor} that stores the pruning modules' weights.

    Attributes:
        scores: A dict {"module_name": Tensor} that stores the scores of pruning modules.
    """

    def __init__(self, modules, config, pattern):
        """Initializes a snip pruning criterion."""
        super(SnipCriterion, self).__init__(modules, config, pattern)
        assert self.config.end_step > 0, "please set end_step > 0 for gradient based criterion"

    def on_before_optimizer_step(self):
        """Calculate and store the pruning scores based on snip criterion."""
        with torch.no_grad():
            for key in self.modules.keys():
                # p = self.modules[key].weight
                param = self.modules[key].weight
                data = safe_get_data(param)
                grad = safe_get_grad(param)
                # self.scores[key] = torch.abs(p * p.grad)
                if hasattr(self.pattern, "reduce_score"):
                    self.scores[key] = self.pattern.reduce_score(torch.abs(data * grad), key)
                else:
                    self.scores[key] = torch.abs(data * grad)


@register_criterion("snip_momentum")
class SnipMomentumCriterion(PruningCriterion):
    """Pruning criterion.

    The snip_momentum criterion_class is derived from PruningCriterion.
    A momentum mechanism is used to calculate snip score, which determines if a weight is to be pruned.

    Args:
        config: A config dict object that includes information about pruner and pruning criterion.
        modules: A dict {"module_name": Tensor} that stores the pruning modules' weights.
        alpha: A parameter that determines how much of the snip score is preserved from last pruning step.
        beta: A parameter that determines how much of the snip score is updated at the current step.

    Attributes:
        scores: A dict {"module_name": Tensor} that stores the scores of pruning modules.
    """

    def __init__(self, modules, config, pattern):
        """Initialize a snip_momentum pruning criterion."""
        super(SnipMomentumCriterion, self).__init__(modules, config, pattern)
        assert self.config.end_step > 0, "please set end_step > 0 for gradient based criterion"
        for key in modules.keys():
            param = modules[key].weight
            # p = modules[key].weight
            param_shape = safe_get_shape(param)
            dtype = torch.float32
            if self.low_memory_usage:
                dtype = torch.bfloat16 if param.device.type == "cpu" else torch.float16
            # self.scores[key] = torch.zeros(p.shape, dtype=dtype).to(p.device)
            if hasattr(self.pattern, "reduce_score"):
                self.scores[key] = self.pattern.reduce_score(
                    torch.zeros(param_shape, dtype=dtype).to(param.device), key
                )
            else:
                self.scores[key] = torch.zeros(param_shape, dtype=dtype).to(param.device)

        self.alpha = 0.9
        self.beta = 1.0

    def on_before_optimizer_step(self):
        """Calculate and store the pruning scores based on snip_momentum criterion."""
        with torch.no_grad():
            for key in self.modules.keys():
                p = self.modules[key].weight
                param = self.modules[key].weight
                data = safe_get_data(param)
                grad = safe_get_grad(param)
                self.scores[key] *= self.alpha
                tmp = torch.abs(data * grad)
                if hasattr(self.pattern, "reduce_score"):
                    tmp = self.pattern.reduce_score(tmp, key, force=True)
                if self.low_memory_usage:
                    tmp = tmp.bfloat16() if p.device.type == "cpu" else tmp.half()
                self.scores[key] += self.beta * tmp


@register_criterion("block_mask")
class BlockMaskCriterion(PruningCriterion):
    """Pruning criterion.

    The block_mask criterion_class is derived from PruningCriterion.
    A momentum mechanism is used to calculate snip score, which determines if a block of weights is to be pruned.

    Args:
        config: A config dict object that includes information about pruner and pruning criterion.
        modules: A dict {"module_name": Tensor} that stores the pruning modules' weights.
        alpha: A parameter that determines how much of the snip score is preserved from last pruning step.
        beta: A parameter that determines how much of the snip score is updated at the current step.

    Attributes:
        scores: A dict {"module_name": Tensor} that stores the scores of pruning modules.
    """

    def __init__(self, modules, config, pattern, masks, alpha=0.9, beta=1.0):
        """Initialize a block_mask pruning criterion."""
        super(BlockMaskCriterion, self).__init__(modules, config, pattern)
        assert self.config.end_step > 0, "please set end_step > 0 for gradient based criterion"
        for key in masks.keys():
            mask = masks[key]
            dtype = torch.float32
            if self.low_memory_usage:
                dtype = torch.bfloat16 if mask.device.type == "cpu" else torch.float16
            self.scores[key] = torch.zeros(mask.shape, dtype=dtype).to(mask.device)
        self.alpha = alpha
        self.beta = beta

    def on_before_optimizer_step(self, masks):
        """Calculate and store the pruning scores based on snip_momentum_block criterion."""
        with torch.no_grad():
            for key in masks.keys():
                grad = masks[key].grad
                if self.low_memory_usage:
                    # TODO check bf16 grad availability
                    grad = grad.bfloat16() if grad.device.type == "cpu" else grad.half()
                self.scores[key] *= self.alpha
                self.scores[key] += self.beta * torch.abs(grad)


@register_criterion("retrain_free")
class RetrainFreeCriterion(PruningCriterion):
    """Pruning criterion.

    The retrain_free criterion_class is derived from PruningCriterion.

    Args:
        config: A config dict object that includes information about pruner and pruning criterion.
        modules: A dict {"module_name": Tensor} that stores the pruning modules' weights.
        alpha: A parameter that determines how much of the snip score is preserved from last pruning step.
        beta: A parameter that determines how much of the snip score is updated at the current step.

    Attributes:
        scores: A dict {"module_name": Tensor} that stores the scores of pruning modules.
    """

    def __init__(self, modules, config, pattern, masks):
        """Initialize a block_mask pruning criterion."""
        super(RetrainFreeCriterion, self).__init__(modules, config, pattern)
        assert self.config.end_step > 0, "please set end_step > 0 for gradient based criterion"
        self.collected_grads = {}
        for key in self.modules.keys():
            for name, param in self.modules[key].named_parameters():
                param.requires_grad_(False)  # only for retrain-free criterion

            if key not in masks.keys():
                continue  # No corresponding block mask, skip.
            mask = masks[key]
            dtype = torch.float32
            if self.low_memory_usage:
                dtype = torch.bfloat16 if mask.device.type == "cpu" else torch.float16
            self.scores[key] = torch.zeros(mask.shape, dtype=dtype).to(mask.device)
            self.collected_grads[key] = []

    def on_before_optimizer_step(self, masks):
        """Calculate and store the pruning scores based on snip_momentum_block criterion."""
        with torch.no_grad():
            for key in masks.keys():
                mask_grad = masks[key].grad.clone()
                if self.low_memory_usage:
                    mask_grad = mask_grad.bfloat16() if mask_grad.device.type == "cpu" else mask_grad.half()
                self.collected_grads[key].append(mask_grad.cpu())
                self.scores[key] += mask_grad.pow(2)
