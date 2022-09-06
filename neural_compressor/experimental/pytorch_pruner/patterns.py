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
    """Register a pattern to the registry"""

    def register(pattern):
        PATTERNS[name] = pattern
        return pattern

    return register


def get_pattern(config):
    """Get registered pattern class"""
    name = config.pattern
    name = name.split('_')[-1]
    if "x" in name:
        return PATTERNS["NxM"](config)
    if ":" in name:
        return PATTERNS["N:M"](config)
    assert False, f"currently only support {PATTERNS.keys()}"


class Pattern:
    def __init__(self, config):
        self.pattern = config.pattern
        self.is_global = config.prune_domain == "global"

    def get_masks(self, scores, target_sparsity_ratio, pre_masks, max_sparsity_ratio_per_layer):
        if self.is_global:
            return self.get_masks_global(scores, target_sparsity_ratio, pre_masks, max_sparsity_ratio_per_layer)
        else:
            return self.get_masks_local(scores, target_sparsity_ratio, pre_masks, max_sparsity_ratio_per_layer)

    def get_masks_global(self, scores, target_sparsity_ratio, pre_masks, max_sparsity_ratio_per_layer):
        raise NotImplementedError

    def get_mask_single(self, score, exact_sparsity_ratio):
        flattern_score = torch.flatten(score)
        k = int(exact_sparsity_ratio * flattern_score.numel())
        threshold, _ = torch.kthvalue(flattern_score, k)
        if not k < 1:
            zero = torch.tensor([0.]).to(score.device)
            one = torch.tensor([1.]).to(score.device)
            mask = torch.where(score <= threshold, zero, one)
        else:
            mask = torch.ones(score.shape,device=score.device)
        return mask


    def get_masks_local(self, scores, target_sparsity_ratio, pre_masks, max_sparsity_ratio_per_layer):
        masks = {}
        for key in scores.keys():
            score = {key: scores[key]}
            pre_mask = {key: pre_masks[key]}
            mask = self.get_masks_global(score, target_sparsity_ratio, pre_mask, max_sparsity_ratio_per_layer)
            masks[key] = mask[key]
        return masks

    def get_sparsity_ratio(self, pre_masks):
        zero_cnt = 0
        total_cnt = 0
        for key in pre_masks.keys():
            pre_mask = pre_masks[key]
            zero_cnt += torch.sum(pre_mask == 0.0).data.item()
            total_cnt += pre_masks.numel()
        return float(zero_cnt) / total_cnt


@register_pattern('NxM')
class PatternNxM(Pattern):
    def __init__(self, config):
        super(PatternNxM, self).__init__(config)
        pattern = self.pattern.split('_')[-1]
        self.block_size = [int(pattern.split('x')[0]), int(pattern.split('x')[1])]

    def get_sparsity_ratio(self, pre_masks):
        zero_cnt = 0
        total_cnt = 0
        block_size = self.block_size
        for key in pre_masks.keys():
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

    def get_masks_global(self, scores, target_sparsity_ratio, pre_masks, max_sparsity_ratio_per_layer):
        block_size = self.block_size
        new_scores = {}
        not_divided_keys = []
        for key in scores.keys():
            current_score = scores[key]
            shape = current_score.shape
            if len(shape) == 4:  ##default is conv, is transpose conv ok ?
                shape = current_score.reshape(current_score.shape[0], -1).shape
            if shape[0] % block_size[0] != 0 or shape[1] % block_size[1] != 0:

                not_divided_keys.append(key)
                continue

            new_shape = [shape[0] // block_size[0], block_size[0], shape[1] // block_size[1], block_size[1]]
            current_score = current_score.reshape(new_shape)
            current_score_sum = current_score.sum(-1).sum(1)
            new_scores[key] = current_score_sum
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
                mask = mask.repeat_interleave(block_size[0], dim=0).repeat_interleave(block_size[1], dim=-1)
                if torch.sum(mask) / mask.numel() < 1.0 - max_sparsity_ratio_per_layer:  
                    ##to prevent some layer not be purned too much
                    ##this is differnt with our original implementation
                    masks[key]=self.get_mask_single(new_scores[key],max_sparsity_ratio_per_layer)
                    masks[key]=masks[key].repeat_interleave(block_size[0], 0).repeat_interleave(block_size[1], -1)
                    # if pre_masks != {}:##when use one shot, this is not right
                    #     masks[key] = pre_masks[key]
                    # else:
                    #     masks[key] = mask
                else:
                    masks[key] = mask
                if len(scores[key].shape)==4:
                    ##we need to revert back
                    masks[key]=masks[key].reshape(scores[key].shape)

            for key in not_divided_keys:
                p = scores[key]
                masks[key] = torch.ones(p.shape).to(p.device)
                logger.warning(f"{key} shape {scores[key].shape} cannot be divided by {self.pattern}")

        else:
            for key in scores.keys():
                p = scores[key]
                masks[key] = torch.ones(p.shape).to(p.device)
        return masks


@register_pattern('N:M')
class PatternNInM(Pattern):
    def __init__(self, config):
        super(PatternNInM, self).__init__(config)
        pattern = self.pattern.split('_')[-1]
        self.N = int(pattern.split(':')[0])
        self.M = int(pattern.split(':')[1])  ##m is bigger

    def get_sparsity_ratio(self, pre_masks):
        ##simply use elemwise sparsity
        non_zero_cnt = 0
        total_cnt = 0
        for key in pre_masks.keys():
            non_zero_cnt += (torch.sum(pre_masks[key])).data.item()
            total_cnt += pre_masks[key].numel()
        return 1.0-float(non_zero_cnt) / total_cnt

    def get_masks_global(self, scores, target_sparsity_ratio, pre_masks, max_sparsity_ratio_per_layer):
        N = self.N
        M = self.M
        target_sparsity_ratio = target_sparsity_ratio / (float(N / M))  ##recover sparsity for block wise
        all_nm_masks = {}
        new_scores = {}
        not_divided_keys = []
        for key in scores.keys():
            current_score = scores[key]
            if len(current_score.shape) == 4:  ##TODO need to verify whether it's ok for transposed conv
                current_score = current_score.permute(0, 2, 3, 1)##cout,k,k,cin
                current_score = current_score.reshape(current_score.shape[0], -1)
            shape = current_score.shape
            new_shape = [shape[0], shape[1] // M, M]
            current_score_new = current_score.reshape(new_shape)

            threshold, _ = torch.kthvalue(current_score_new, N, dim=2)
            threshold = threshold.unsqueeze(-1)
            if shape[1] % M != 0:
                not_divided_keys.append(key)
                continue
            threshold = threshold.expand(shape[0], shape[1] // M, M)
            threshold = threshold.reshape((shape[0], shape[1]))

            one = torch.tensor([1.]).to(current_score.device)
            zero = torch.tensor([0.]).to(current_score.device)
            mask = torch.where(current_score <= threshold, zero, one)
            current_score_new = current_score_new.reshape((shape[0], shape[1]))
            ##to get the sum of N scores in each block with M
            current_score_new = current_score_new * (1.0 - mask)
            current_score_new = current_score_new.reshape(shape[0], shape[1] // M, M)
            score_sum = torch.sum(current_score_new, dim=-1)
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
