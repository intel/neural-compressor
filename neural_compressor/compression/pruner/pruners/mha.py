"""Mha pruner."""

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

from ..criteria import get_criterion
from ..patterns import get_pattern
from ..schedulers import get_scheduler
from ..utils import logger, torch
from .base import PytorchBasePruner, register_pruner


@register_pruner("pt_mha")
class PythonMultiheadAttentionPruner(PytorchBasePruner):
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
        mha_compressions: a Dict. (key: MHA module name; value: MHACompression object in .model_slim.weight_slim)
            Main object to hook critical attributes for mha pruning and modify these attributes.
        linear_layers: a Dict. {key: linear layer name; value: torch.nn.Linear object.}
            Store independent linear layer look-up table, which used by criterion object.
            linear_layers length should be 4x of mha_compression because one mha_compression hooks 4 linear layers:
            query, key, value and subsequent ffn layer.
        head_masks: A dict. {key: MHA module name; value: torch.Tensor(1, mha_head_size)}
            Similar to Huggingface built-in head_mask attribute.
        mha_scores: A dict. {key: MHA module name; value: torch.Tensor(1, mha_head_size)}
            Store scores for different heads.
    """

    def __init__(self, config, mha_modules):
        """Initialize."""
        # use pattern search technique to obtain multihead attention modules
        # modules is a dict that fits the mha auto slim process
        # -----------------------------------------
        self.config = config
        self.mha_modules = mha_modules
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
        self.criterion_reduce_type = self.config["criterion_reduce_type"]
        self.pruning_scope = self.config["pruning_scope"]
        # ------------------------Custom attributes for MHA Pruner--------------------------------------
        # main initialize process.
        # define some attributes.
        self.mha_compressions = {}
        self.linear_layers = {}
        self.head_masks = {}
        self.mha_scores = {}  # {}
        # main initialization process
        self._init_mha_attrs()
        # initialize custom attributes: criterion (snip-momnetum, snip, magnitude, etc.)
        # we have hook modules in mha_compressions therefore do not pass them to patterns
        self.pattern = get_pattern(self.config, modules=None)
        # criterion hooks on linear themselves
        self.criterion = get_criterion(self.config, self.linear_layers, self.pattern)
        self.scheduler = get_scheduler(self.config)
        # -----------------------------------------------------------------------------------------------

    def _init_mha_attrs(self):
        """Initialize self.mha_compressions, self.linear_layers, self.head_masks
        # similar to original mha slim process, but only hook mha modules and their attributes,
        # do not call slim main functions."""
        # auto slim related: head pruning objects
        from ..model_slim.weight_slim import MHACompression

        for mha_module in self.mha_modules:
            # initialize self.mha_compressions
            mha_comp = MHACompression(mha_module)
            self.mha_compressions[mha_module["mha_name"][0]] = mha_comp
            head_nums_for_this_mha = getattr(mha_comp.mha[0], mha_comp.attributes_for_this_mha["head_nums"])
            # initialize head_masks
            # why use 1 x head_num shape? because this provides convenience for permute mask for qkv and ffn
            self.head_masks[mha_module["mha_name"][0]] = torch.ones(1, head_nums_for_this_mha)
            # initialize self.linear_layers
            for idx in range(mha_module["qkv_name"].__len__()):
                # update qkv layers
                self.linear_layers[mha_module["qkv_name"][idx]] = mha_module["qkv_module"][idx]
            for idx in range(mha_module["ffn_name"].__len__()):
                self.linear_layers[mha_module["ffn_name"][idx]] = mha_module["ffn_module"][idx]

    def reduce_mha_scores(self, score, dim=0):
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
            head_size = getattr(mha_comp.mha[0], mha_comp.attributes_for_this_mha["head_size"])
            head_nums = getattr(mha_comp.mha[0], mha_comp.attributes_for_this_mha["head_nums"])
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
                qkv_block_size[1],
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
                ffn_block_size[1],
            ]
            for ffn_name, ffn_score in ffn_scores_for_this_mha.items():
                ffn_score_new = ffn_score.reshape(ffn_new_shape)
                ffn_score_new = self.reduce_mha_scores(self.reduce_mha_scores(ffn_score_new, -1), 1)
                # ffn_scores_for_this_mha[ffn_name] = ffn_score_new # [1, head_nums]
                ffn_gather_scores += ffn_score_new
            # step 3: compile qkv ffn scores to obtain individual head's score
            self.mha_scores[mha_name] = qkv_gather_scores + ffn_gather_scores.permute(1, 0)
            self.mha_scores[mha_name] /= len(qkv_scores_for_this_mha) + len(ffn_scores_for_this_mha)  # should be 4
        return True

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
            self.head_masks,
            self.init_sparsity_ratio,
        )
        logger.info(f"current target ratio is {current_target_sparsity_ratio}")

        self.completed_pruned_cnt += 1
        if self.criterion.scores == {}:
            return

        # self.masks = self.pattern.get_masks(self.criterion.scores, current_target_sparsity_ratio, self.masks)
        # self.mask_weights()
        self.update_mha_scores()  # update self.mha_scores
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
