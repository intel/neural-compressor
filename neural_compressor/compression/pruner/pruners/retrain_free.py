"""Retrain free pruner."""

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

from functools import partial

from ..criteria import get_criterion
from ..patterns import get_pattern
from ..regs import get_reg
from ..schedulers import get_scheduler
from ..utils import F, logger, torch
from .base import PytorchBasePruner, register_pruner


@register_pruner("pt_retrain_free")
class PytorchRetrainFreePruner(PytorchBasePruner):
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

    def __init__(self, config, modules):
        """Initialize."""
        super().__init__(config, modules)

    def _init(self):
        """Initialize."""
        self.pattern = get_pattern(self.config, self.modules)
        self.masks = self.pattern.register_block_masks()
        self.pruner_masks = [self.masks]
        self._rewrite_forward(self.pruner_masks)
        self.scheduler = get_scheduler(self.config)
        self.criterion = get_criterion(modules=self.modules, config=self.config, pattern=self.pattern, masks=self.masks)
        self.reg = get_reg(self.config, self.modules, self.pattern)
        logger.warning("Retrain-free pruner fixed the weights, please DO NOT turn on gradient update.")
        assert (
            "channel" in self.pattern.pattern
        ), "retrain-free pruner only supports large patterns like channel-wise pruning."

    def _rewrite_forward(self, pruner_masks):
        def forward(self, input):
            block_mask = pruner_masks[0][self.mask_name]
            block_mask.requires_grad_(True)  # Makesure that the gradient of block mask is always avilible
            block_size = [self.weight.shape[0] // block_mask.shape[0], self.weight.shape[1] // block_mask.shape[1]]
            mask = (
                block_mask.repeat_interleave(block_size[0], dim=0)
                .repeat_interleave(block_size[1], dim=-1)
                .to(self.weight.device)
            )
            return F.linear(input, self.weight * mask, self.bias)

        for key in self.masks.keys():
            module = self.modules[key]
            module.mask_name = key
            module.forward = partial(forward, module)

    def _recover_forward(self):
        with torch.no_grad():
            for key in self.masks.keys():
                module = self.modules[key]
                delattr(module, "mask_name")
                self.masks[key].requires_grad_(False)
                module.forward = partial(torch.nn.Linear.forward, module)

    # def on_step_begin(self, local_step):
    #     """Implement at the start of each step.

    #     Update the masks at a given local_step.
    #     """
    #     self.update_masks(local_step)

    def update_masks(self, local_step):
        """Update the masks at a given local step."""
        if self.global_step == self.start_step:
            if self.config["lock_init_sparsity"]:
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
        # the order of the following three lines can't not be exchanged
        self.masks = self.pattern.get_masks(self.criterion.scores, current_target_sparsity_ratio, self.masks)
        self.masks = self.rearrange_masks(self.masks)
        self.pruner_masks[0] = self.masks

        self.current_sparsity_ratio = self.pattern.get_sparsity_ratio(self.masks)
        logger.info(f"current sparsity ratio is {self.current_sparsity_ratio}")

    def on_step_end(self):
        """Implement after every step."""
        if self.global_step >= self.start_step and self.global_step <= self.end_step:
            self.criterion.on_before_optimizer_step(self.masks)
        self.zero_mask_grad()
        if self.end_step == self.global_step:
            self.mask_weights()
            logger.info(f"mask weights at last_prune_step: {self.global_step}")
            # recover forward method at last prune step
            self._recover_forward()
        self.global_step += 1

    def mask_weights(self):
        """Apply block masks to corresponding modules' weights.

        Weights are multiplied with masks. This is the formal pruning process.
        """
        with torch.no_grad():
            self.pattern.mask_block_weights(self.masks)

    def rearrange_masks(self, masks):
        """Rearrange the masks of each layer with constant sparsity."""
        with torch.no_grad():
            for key in masks.keys():
                block_mask = masks[key]
                num_pruned = torch.sum(block_mask == 0.0).data.item()
                if not num_pruned or not self.criterion.collected_grads[key]:
                    continue
                grads = torch.stack(self.criterion.collected_grads[key], dim=0).squeeze()
                # self.criterion.collected_grads[key] = [] # clear grads after each pruning step
                grads = grads.permute(1, 0).contiguous()
                grads_sq = grads.pow(2).sum(dim=1)
                _, indices = grads_sq.sort(descending=False)
                indices = indices.tolist()
                masked_indices = indices[:num_pruned]
                for index in indices[num_pruned:]:
                    masked_indices.append(index)
                    grad_vectors = grads[masked_indices]
                    grad_sum = grad_vectors.sum(dim=0)
                    complement = grad_sum - grad_vectors
                    grad_sum_length = complement.pow(2).sum(dim=1)
                    removed = grad_sum_length.argmin()
                    del masked_indices[removed]

                new_mask = torch.ones(len(indices)).to(block_mask.device)
                new_mask[masked_indices] = 0
                new_mask = new_mask * torch.ones_like(block_mask, device=block_mask.device, dtype=block_mask.dtype)
                block_mask.data = new_mask.data
        return masks

    def zero_mask_grad(self):
        with torch.no_grad():
            for key in self.masks.keys():
                mask = self.masks[key]
                if mask.grad is not None:
                    if mask.grad.grad_fn is not None:
                        mask.grad.detach_()
                    else:
                        mask.grad.requires_grad_(False)
                    mask.grad.zero_()
