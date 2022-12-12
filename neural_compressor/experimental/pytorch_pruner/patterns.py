"""pattern module."""
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

import logging

import torch
from .logger import logger

PATTERNS = {}


def register_pattern(name):
    """Class decorator used to register a Pattern subclass to the registry.

    Decorator function used before a Pattern subclasses. 
    Make sure that this Pattern class can be registered in PATTERNS.

    Args:
        cls (class): The class of register.
        name: A string. Define the pattern type which will be included in a pruning process.

    Returns:
        cls: The class of register.
    
    """

    def register(pattern):
        PATTERNS[name] = pattern
        return pattern

    return register


def get_pattern(config):
    """Get registered pattern class.

    Get a Pattern object from PATTERNS.

    Args:
        config: A config dict object. Contains the pattern information.

    Returns:
        A Pattern object.

    Raises:
        AssertionError: Currently only support patterns which have been registered in PATTERNS.
    """
    name = config.pattern
    name = name.split('_')[-1]
    if "x" in name:
        return PATTERNS["NxM"](config)
    if ":" in name:
        return PATTERNS["N:M"](config)
    assert False, f"currently only support {PATTERNS.keys()}"


class Pattern:
    """Pruning Pattern.

    Every Pruner object will contain a Pattern object.
    It defines the basic pruning unit and how this unit will be pruned during pruning.

    Args:
        config: A config dict object. Contains the pattern information.

    Attributes:
        pattern: A config dict object. The pattern related part in args config.
        is_global: A bool. Whether the pruning take global pruning option.
            Global pruning means that all pruning layers are gathered to calculate pruning criteria.
            Local pruning, on the contrast, means that pruning layers are to calculate criteria individually.
    """

    def __init__(self, config):
        """Initialize."""
        self.pattern = config.pattern
        self.is_global = config.prune_domain == "global"

    def get_masks(self, scores, target_sparsity_ratio, pre_masks, max_sparsity_ratio_per_layer):
        """Call when new masks for pruning are to be calculated.

        Args:
            scores: A dict{“layer_name”: Tensor}. Store the pruning scores of weights.
            target_sparsity_ratio: A float. After pruning, the model's sparsity will reach this value.
            pre_masks: A dict{"layer_name": Tensor}. The masks generated after the last pruning step.
            max_sparsity_ratio_per_layer: A float. The maximum sparsity that one layer can reach.

        Returns:
            A dict with the identical size as pre_masks. Update the 0/1 values in it. 
        
        """
        if self.is_global:
            return self.get_masks_global(scores, target_sparsity_ratio, pre_masks, max_sparsity_ratio_per_layer)
        else:
            return self.get_masks_local(scores, target_sparsity_ratio, pre_masks, max_sparsity_ratio_per_layer)

    def get_masks_global(self, scores, target_sparsity_ratio, pre_masks, max_sparsity_ratio_per_layer):
        """To be implemented in subclasses."""
        raise NotImplementedError

    def get_mask_single(self, score, exact_sparsity_ratio):
        """Obtain a mask for one layer.

        Args:
            score: A Tensor. Store the pruning scores of one layer.
            exact_sparsity_ratio: A float. After pruning, the layer's sparsity will reach this value.            

        Returns:
            A Tensor with the identical size as score. a new mask.
        """
        flattern_score = torch.flatten(score)
        k = int(exact_sparsity_ratio * flattern_score.numel())
        threshold, _ = torch.kthvalue(flattern_score, k)
        if not k < 1:
            zero = torch.tensor([0.]).to(score.device)
            one = torch.tensor([1.]).to(score.device)
            mask = torch.where(score <= threshold, zero, one)
        else:
            mask = torch.ones(score.shape, device=score.device)
        return mask

    def get_block_size_dict(self, data):
        """To be implemented in subclasses."""
        raise NotImplementedError

    def get_masks_local(self, scores, target_sparsity_ratio, pre_masks, max_sparsity_ratio_per_layer):
        """Obtain layers' local masks.

        Args:
            scores: A dict{“layer_name”: Tensor}. Store the pruning scores of weights.
            target_sparsity_ratio: A float. After pruning, the model's sparsity will reach this value.
            pre_masks: A dict{"layer_name": Tensor}. The masks generated after the last pruning step.
            max_sparsity_ratio_per_layer: A float. The maximum sparsity that one layer can reach.

        Returns:
            A dict with the identical size as pre_masks. Update the 0/1 values in it. 
        """
        masks = {}
        if isinstance(self, PatternNxM) and not isinstance(self.block_size, dict):
            self.block_size = self.get_block_size_dict(pre_masks)
        for key in scores.keys():
            score = {key: scores[key]}
            pre_mask = {key: pre_masks[key]}
            mask = self.get_masks_global(score, target_sparsity_ratio, pre_mask, max_sparsity_ratio_per_layer)
            masks[key] = mask[key]
        return masks

    def get_sparsity_ratio(self, pre_masks):
        """Calulate the zero elements' ration in pre_masks.
        
        Args:
            pre_masks: Dict{"layer_name": Tensor}. The masks generated after the last pruning step.
        
        Returns:
            A float. The zero elements' ratio in pre_masks.
        """
        zero_cnt = 0
        total_cnt = 0
        for key in pre_masks.keys():
            pre_mask = pre_masks[key]
            zero_cnt += torch.sum(pre_mask == 0.0).data.item()
            total_cnt += pre_masks.numel()
        return float(zero_cnt) / total_cnt

    def get_pattern_lock_masks(self, modules):
        """Obtain masks from original weight map, by masking where weights' are zero.

        Args:
            modules: A dict{“layer_name”: Tensor}. Store weights.

        Returns:
            A dict with the identical size as modules, containing pattern lock masks.
        """
        pattern_lock_masks = {}
        for key in modules.keys():
            weight = modules[key].weight
            shape = weight.shape
            mask = torch.ones(shape)
            mask[weight == 0] = 0.0
            pattern_lock_masks[key] = mask.to(weight.device)
        return pattern_lock_masks


@register_pattern('NxM')
class PatternNxM(Pattern):
    """Pruning Pattern.

    A Pattern class derived from Pattern. In this pattern, the weights in a NxM block will be pruned or kept
    during one pruning step.

    Args:
        config: A config dict object. Contains the pattern information.

    Attributes:
        block_size: A list of two Integers. The height and width of the block.
            Please be aware that the vertical direction of a Linear layer's weight in PyTorch refer to output channel.
            Because PyTorch's tensor matmul has a hidden transpose operation.
    """

    def __init__(self, config):
        """Initialize."""
        super(PatternNxM, self).__init__(config)
        pattern = self.pattern.split('_')[-1]
        self.N = pattern.split('x')[0]
        self.M = pattern.split('x')[1]
        if self.N == "channel":  ##channel-wise pruning mode
            self.block_size = ["channel", int(self.M)]
        elif self.M == "channel":  ##channel-wise pruning mode
            self.block_size = [int(self.N), "channel"]
        else:
            self.block_size = [int(pattern.split('x')[0]), int(pattern.split('x')[1])]

    def get_block_size_dict(self, data):
        """Calulate the zero elements' ration in pre_masks.
        
        Args:
            data: Dict{"layer_name": Tensor}. Store weights or scores.
        
        Returns:
            A dict. Dict{"layer_name": [block_size_1, block_size_2]}.
                Containing layers' corresponding pruning pattern's block shape.
                Please be aware that because in channel-wise pruning, 
                different layers can have different pruning patterns. 
        """
        block_sizes_dict = {}
        if self.N == "channel" or self.M == "channel":
            for key in data.keys():
                if isinstance(data[key], torch.nn.Module):
                    shape = data[key].weight.shape
                else:
                    shape = data[key].shape
                if self.N == "channel":
                    block_sizes_dict[key] = [shape[0], 1]
                else:
                    block_sizes_dict[key] = [1, shape[1]]
            return block_sizes_dict
        for key in data.keys():
            block_sizes_dict[key] = self.block_size
        return block_sizes_dict

    def get_sparsity_ratio(self, pre_masks):
        """Calulate the zero elements' ration in pre_masks.
        
        Args:
            pre_masks: Dict{"layer_name": Tensor}. The masks generated after the last pruning step.
        
        Returns:
            A float. Calculate the zero elements' ratio in pre_masks.
        """
        zero_cnt = 0
        total_cnt = 0
        if isinstance(self.block_size, list):
            self.block_size = self.get_block_size_dict(pre_masks)
        for key in pre_masks.keys():
            block_size = self.block_size[key]
            pre_mask = pre_masks[key]
            shape = pre_mask.shape
            if len(shape) == 4:
                shape = pre_mask.reshape(pre_mask.shape[0], -1).shape
            if shape[0] % block_size[0] != 0 or shape[1] % block_size[1] != 0:
                logger.warning(f"layer {key} is not support under current pattern, ignoring")
                continue

            new_shape = [shape[0] // block_size[0], block_size[0], shape[1] // block_size[1], block_size[1]]
            pre_mask = pre_mask.reshape(new_shape)
            pre_mask_sum = pre_mask.sum(-1).sum(1)
            zero_cnt += torch.sum(pre_mask_sum == 0.0).data.item()
            total_cnt += pre_mask_sum.numel()
        return float(zero_cnt) / total_cnt

    def get_masks_global(self, scores, target_sparsity_ratio, pre_masks, max_sparsity_ratio_per_layer,
                         keep_pre_mask=False):
        """Generate masks for layers.

        Gather all layer's scores together and calculate a common threshold.
        This threshold will be applied for all layers.

        Args:
            scores: A dict{“layer_name”: Tensor}. Store the pruning scores of weights.
            target_sparsity_ratio: A float. After pruning, the model's sparsity will reach this value.
            pre_masks: A dict{"layer_name": Tensor}. The masks generated after the last pruning step.
            max_sparsity_ratio_per_layer: A float. The maximum sparsity that one layer can reach.
            keep_pre_masks: A bool. If True, keep the masks unchanged.

        Returns:
            A dict with the identical size as pre_masks. Update the 0/1 values in it. 
        """
        if isinstance(self.block_size, list):
            self.block_size = self.get_block_size_dict(scores)
        new_scores = {}
        not_divided_keys = []
        for key in scores.keys():
            block_size = self.block_size[key]
            current_score = scores[key]
            if len(current_score.shape) == 4:  ##TODO need to verify whether it's ok for transposed conv
                current_score = current_score.permute(0, 2, 3, 1)  ##cout,k,k,cin
                current_score = current_score.reshape(current_score.shape[0], -1)
            shape = current_score.shape
            if shape[0] % block_size[0] != 0 or shape[1] % block_size[1] != 0:  ## only consider input channel
                not_divided_keys.append(key)
                continue

            new_shape = [shape[0] // block_size[0], block_size[0], shape[1] // block_size[1],
                         block_size[1]]
            current_score = current_score.reshape(new_shape)
            current_score_sum = current_score.mean(-1).mean(
                1)  ##TODO sum or mean is quite different for per channel pruning
            new_scores[key] = current_score_sum
        global_scores = torch.cat([torch.flatten(v) for v in new_scores.values()])
        k = int(target_sparsity_ratio * global_scores.numel())
        masks = {}
        if not k < 1:
            threshold, _ = torch.kthvalue(global_scores, k)
            for key in new_scores.keys():
                block_size = self.block_size[key]
                score = new_scores[key]
                zero = torch.tensor([0.]).to(score.device)
                one = torch.tensor([1.]).to(score.device)
                mask = torch.where(score <= threshold, zero, one)
                mask = mask.repeat_interleave(block_size[0], dim=0).repeat_interleave(block_size[1], dim=-1)
                if torch.sum(mask) / mask.numel() < 1.0 - max_sparsity_ratio_per_layer:
                    ##to prevent some layer not be purned too much
                    ##this is differnt with our original implementation
                    masks[key] = self.get_mask_single(new_scores[key], max_sparsity_ratio_per_layer)
                    masks[key] = masks[key].repeat_interleave(block_size[0], 0).repeat_interleave(block_size[1], -1)
                    # if pre_masks != {}:##when use one shot, this is not right
                    #     masks[key] = pre_masks[key]
                    # else:
                    #     masks[key] = mask
                else:
                    masks[key] = mask
                # if len(scores[key].shape) == 4:
                #     ##we need to revert back
                #     masks[key] = masks[key].reshape(scores[key].shape)

            for key in not_divided_keys:
                p = scores[key]
                masks[key] = torch.ones(p.shape).to(p.device)
                logger.warning(f"{key} shape {scores[key].shape} cannot be divided by {self.pattern}")

        else:
            for key in scores.keys():
                p = scores[key]
                masks[key] = torch.ones(p.shape).to(p.device)

        for key in masks.keys():
            if len(scores[key].shape) == 4 and len(masks[key].shape) == 2:  ## need to permute
                mask = masks[key]
                mask = mask.reshape(scores[key].shape[0], scores[key].shape[2], scores[key].shape[3],
                                    scores[key].shape[1])
                mask = mask.permute(0, 3, 1, 2)
                masks[key] = mask
        return masks

    def get_pattern_lock_masks(self, modules):
        """Obtain masks from original weight map, by masking where weights' are zero.

        Args:
            modules: A dict{“layer_name”: Tensor}. Store weights.

        Returns:
            A dict with the identical size as modules, containing pattern lock masks.
        """
        pattern_lock_masks = {}
        if isinstance(self.block_size, list):
            self.block_size = self.get_block_size_dict(modules)
        for key in modules.keys():
            block_size = self.block_size[key]
            weight = modules[key].weight
            if len(weight.shape) == 4:  # conv
                weight = weight.permute(0, 2, 3, 1)
                weight = weight.reshape(weight.shape[0], -1)
            shape = weight.shape
            new_shape = [shape[0] // block_size[0], block_size[0], shape[1] // block_size[1], block_size[1]]
            p = weight.reshape(new_shape)
            p_mag = p.abs()  # avoid the scene which sum is zero but weights are not
            weight_block_sum = p_mag.sum(-1).sum(1)
            mask = torch.ones(weight_block_sum.shape)
            mask[weight_block_sum == 0] = 0.0
            mask = mask.repeat_interleave(block_size[0], dim=0).repeat_interleave(block_size[1], dim=-1)
            orig_shape = modules[key].weight.shape
            if len(orig_shape) == 4:
                mask = mask.reshape(orig_shape[0], orig_shape[2], orig_shape[3], orig_shape[1])
                mask = mask.permute(0, 3, 1, 2)
            pattern_lock_masks[key] = mask.to(weight.device)
        return pattern_lock_masks


@register_pattern('N:M')
class PatternNInM(Pattern):
    """Pruning Pattern.

    A Pattern class derived from Pattern. In this pattern, N out of every M continuous weights will be pruned.
    For more info of this pattern, please refer to 
    https://github.com/intel/neural-compressor/blob/master/docs/pruning.md

    Args:
        config: A config dict object. Contains the pattern information.

    Attributes:
        N: The number of elements to be prune in a weight sequence.
        M: The size of the weight sequence.

    """

    def __init__(self, config):
        """Initialize."""
        super(PatternNInM, self).__init__(config)
        pattern = self.pattern.split('_')[-1]
        self.N = int(pattern.split(':')[0])
        self.M = int(pattern.split(':')[1])  ##m is bigger

    def get_sparsity_ratio(self, pre_masks):
        """Calulate the zero elements' ration in pre_masks.
        
        Args:
            pre_masks: Dict{"layer_name": Tensor}. The masks generated after the last pruning step.
        
        Returns:
            A float. Calculate the zero elements' ratio in pre_masks.
        """
        ##simply use elemwise sparsity
        non_zero_cnt = 0
        total_cnt = 0
        for key in pre_masks.keys():
            non_zero_cnt += (torch.sum(pre_masks[key])).data.item()
            total_cnt += pre_masks[key].numel()
        return 1.0 - float(non_zero_cnt) / total_cnt

    def get_masks_global(self, scores, target_sparsity_ratio, pre_masks, max_sparsity_ratio_per_layer):
        """Generate masks for layers.

        Gather all layer's scores together and calculate a common threshold.
        This threshold will be applied for all layers.

        Args:
            scores: A dict{“layer_name”: Tensor}. Store the pruning scores of weights.
            target_sparsity_ratio: A float. After pruning, the model's sparsity will reach this value.
            pre_masks: A dict{"layer_name": Tensor}. The masks generated after the last pruning step.
            max_sparsity_ratio_per_layer: A float. The maximum sparsity that one layer can reach.

        Returns:
            A dict with the identical size as pre_masks. Update the 0/1 values in it. 
        """
        N = self.N
        M = self.M
        target_sparsity_ratio = target_sparsity_ratio / (float(N / M))  ##recover sparsity for block wise
        all_nm_masks = {}
        new_scores = {}
        not_divided_keys = []
        for key in scores.keys():
            current_score = scores[key]
            shape = current_score.shape
            if shape[1] % M != 0:
                not_divided_keys.append(key)
                continue
            if len(current_score.shape) == 4:  ##TODO need to verify whether it's ok for transposed conv
                current_score = current_score.permute(0, 2, 3, 1)  ##cout,k,k,cin
                current_score = current_score.reshape(current_score.shape[0], -1)
                shape = current_score.shape
            new_shape = [shape[0], shape[1] // M, M]
            current_score_new = current_score.reshape(new_shape)

            threshold, _ = torch.kthvalue(current_score_new, N, dim=2)
            threshold = threshold.unsqueeze(-1)

            threshold = threshold.expand(shape[0], shape[1] // M, M)
            threshold = threshold.reshape((shape[0], shape[1]))

            one = torch.tensor([1.]).to(current_score.device)
            zero = torch.tensor([0.]).to(current_score.device)
            mask = torch.where(current_score <= threshold, zero, one)
            current_score_new = current_score_new.reshape((shape[0], shape[1]))
            ##to get the sum of N scores in each block with M
            current_score_new = current_score_new * (1.0 - mask)
            current_score_new = current_score_new.reshape(shape[0], shape[1] // M, M)
            score_sum = torch.mean(current_score_new, dim=-1)
            all_nm_masks[key] = mask
            new_scores[key] = score_sum

        global_scores = torch.cat([torch.flatten(v) for v in new_scores.values()])
        k = int(target_sparsity_ratio * global_scores.numel())
        masks = {}
        if not k < 1:
            threshold, _ = torch.kthvalue(global_scores, k)
            for key in new_scores.keys():
                score = new_scores[key]
                zero = torch.tensor([0.]).to(score.device)
                one = torch.tensor([1.]).to(score.device)
                mask = torch.where(score <= threshold, zero, one)
                mask = mask.repeat_interleave(M, dim=-1)
                ## both zero will be zero
                mask = (mask + all_nm_masks[key])
                mask = torch.where(mask <= 0, zero, one)
                if torch.sum(mask) / mask.numel() < 1.0 - max_sparsity_ratio_per_layer:
                    ##trick, to prevent some layer not be purned too much
                    masks[key] = self.get_mask_single(new_scores[key], max_sparsity_ratio_per_layer)
                    masks[key] = masks[key].repeat_interleave(M, dim=-1)
                    ## both zero will be zero
                    masks[key] = (masks[key] + all_nm_masks[key])
                    masks[key] = torch.where(masks[key] <= 0, zero, one)
                else:
                    masks[key] = mask
            for key in not_divided_keys:
                p = scores[key]
                masks[key] = torch.ones(p.shape).to(p.device)
                logger.warning(f"{key} shape {scores[key].shape} cannot be divided by {self.pattern}")

        else:
            for key in scores.keys():
                p = scores[key]
                masks[key] = torch.ones(p.shape).to(p.device)
        for key in masks.keys():
            if len(scores[key].shape) == 4 and len(masks[key].shape) == 2:  ## need to permute
                mask = masks[key]
                mask = mask.reshape(scores[key].shape[0], scores[key].shape[2], scores[key].shape[3],
                                    scores[key].shape[1])
                mask = mask.permute(0, 3, 1, 2)
                masks[key] = mask

        return masks

    def get_pattern_lock_masks(self, modules):
        """Obtain masks from original weight map, by masking where weights' are zero.

        Args:
            modules: A dict{“layer_name”: Tensor}. Store weights.

        Returns:
            A dict with the identical size as modules, containing pattern lock masks.
        """
        pattern_lock_masks = {}
        N, M = self.N, self.M
        for key in modules.keys():
            weight = modules[key].weight
            if len(weight.shape) == 4:  # conv
                weight = weight.permute(0, 2, 3, 1)
                weight = weight.reshape(weight.shape[0], -1)
            shape = weight.shape
            ##TODO need to check whether it can be divisible later
            new_shape = [shape[0], shape[1] // M, M]
            weight_new = weight.reshape(new_shape)
            mask1 = torch.ones(weight_new.shape)
            mask2 = torch.ones(weight_new.shape)
            nonzeros = torch.count_nonzero(weight_new, dim=-1)
            zeros = M - nonzeros
            mask1[weight_new == 0] = 0.0
            mask2[zeros >= N] = 0.0
            mask3 = mask1 + mask2  # zero in mask3 means its block has been completely pruned.
            zero = torch.tensor([0.]).to(weight.device)
            one = torch.tensor([1.]).to(weight.device)
            mask = torch.where(mask3 == 0, zero, one)
            mask = mask.reshape(shape)
            orig_shape = modules[key].weight.shape
            if len(orig_shape) == 4:
                mask = mask.reshape(orig_shape[0], orig_shape[2], orig_shape[3], orig_shape[1])
                mask = mask.permute(0, 3, 1, 2)

            pattern_lock_masks[key] = mask.to(weight.device)
        return pattern_lock_masks
