"""MHA patterns."""

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
from ..utils import torch
from .base import PytorchBasePattern, register_pattern


@register_pattern("ptMHA")
class PatternMHA(PytorchBasePattern):
    """Pruning Pattern.

    A Pattern class derived from BasePattern. In this pattern, we calculate head masks for a MHA module
    For more info of this pattern, please refer to :
    https://github.com/intel/neural-compressor/blob/master/docs/sparsity.md

    Args:
        config: A config dict object that contains the pattern information.

    Attributes:
        N: The number of elements to be pruned in a weight sequence.
        M: The size of the weight sequence.
    """

    def __init__(self, config, modules=None):
        self.is_global = config.pruning_scope == "global"

    # only implement three method: get_masks, get_masks_local, get_masks_global

    def get_masks_global(self, scores, target_sparsity_ratio, pre_masks):
        # gather all score items into one tensor
        if target_sparsity_ratio <= 0.0:
            return pre_masks
        flatten_score = torch.cat(list(scores.values())).flatten()
        k = int(target_sparsity_ratio * flatten_score.numel())
        if k <= 0:
            return pre_masks
        threshold, _ = torch.kthvalue(flatten_score, k)
        head_masks = {}
        zero = torch.tensor([0.0]).to(threshold.device)
        one = torch.tensor([1.0]).to(threshold.device)
        for mha_name, mha_score in scores.items():
            head_masks[mha_name] = torch.where(mha_score <= threshold, zero, one).permute(1, 0)
            head_masks[mha_name] = head_masks[mha_name].bool()
        return head_masks
