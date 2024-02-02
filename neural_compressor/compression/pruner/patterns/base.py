"""Pruning patterns."""

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

from collections import namedtuple

import numpy as np

from ..utils import safe_get_data, safe_get_grad, safe_get_shape, tf, torch

PATTERNS = {}


def register_pattern(name):
    """Class decorator used to register a Pattern subclass to the registry.

    Decorator function used before a Pattern subclasses.
    Make sure that this Pattern class can be registered in PATTERNS.

    Args:
        name: A string defining the pattern type name to be used in a pruning process.

    Returns:
        cls: The class of register.
    """

    def register(pattern):
        """Register patterns."""
        PATTERNS[name] = pattern
        return pattern

    return register


SparsityInfo = namedtuple("SparsityInfo", ["zero_cnt", "total_cnt", "sparsity_ratio"])


class ProgressivePatternUtils(object):
    @staticmethod
    def _reshape_orig_to_2dims(data):
        """Process layers that are not two-dimensional(e.g conv layer).

        Args:
            data: Input.

        Returns:
            Reshaped data.
        """
        if len(data.shape) == 4:  ##TODO need to verify whether it's ok for transposed conv
            data = data.permute(0, 2, 3, 1)  ##cout,k,k,cin
            data = data.reshape(data.shape[0], -1)
        return data

    @staticmethod
    def _reshape_2dims_to_orig(data, orig_shape):
        """Recover layers that are not two-dimensional(e.g conv layer).

        Args:
            data: Input.

        Returns:
            Reshaped data.
        """
        if len(orig_shape) == 2:
            return data
        elif len(orig_shape) == 4:
            data = data.reshape(orig_shape[0], orig_shape[2], orig_shape[3], orig_shape[1])
            data = data.permute(0, 3, 1, 2)
        elif len(orig_shape) == 3:
            data = data.reshape(orig_shape[0], orig_shape[2], orig_shape[1])
            data = data.permute(0, 2, 1)
        elif len(orig_shape) == 1:
            data = data.reshape(orig_shape)
        else:
            raise NotImplementedError(f"not support {data.shape}")
        return data

    # some util functions which can be used.
    @staticmethod
    def count_new_masked_cnts(new_added_masks):
        """Count the number of elements to be masked.

        Args:
            new_added_masks: A dict {"layer_name": Tensor} that stores the added masks.

        Returns:
            The number of masked weights.
        """
        # count how many elements are to masked,
        new_masked_cnts = 0
        for key in new_added_masks.keys():
            new_masked_cnts += torch.nonzero(1 - new_added_masks[key]).size()[0]
        return new_masked_cnts

    @staticmethod
    def update_new_added_masks(pre_masks, cur_masks):
        """Obtain the new set-to-zero masks during a pruning procedure.

        Pre_masks, cur_masks should have identical keys because they represent the same model.

        Args:
            pre_masks: Dict{"layer_name": Tensor} that stores the masks generated after the last pruning step.
            cur_masks: Dict{"layer_name": Tensor} that stores the current masks.

        Returns:
            A dict {"layer_name": Tensor} that stores the added masks.
        """
        # obtain the new set-to-zero mask during a pruning procedure.
        # pre_masks, cur_masks should have identical keys because they stands for one model.
        new_added_masks = {}
        for key in pre_masks.keys():
            pre_mask = pre_masks[key]
            cur_mask = cur_masks[key]
            zero = torch.tensor([0.0]).to(pre_mask.device)
            one = torch.tensor([1.0]).to(cur_mask.device)
            new_added_masks[key] = torch.where(pre_mask == cur_mask, one, zero)
        return new_added_masks

    @staticmethod
    def update_progressive_masks_global_scores(pre_masks, cur_masks, scores, progressive_step, progressive_configs):
        """Generate the progressive masks.

        Args:
            pre_masks: Dict{"layer_name": Tensor} that stores the masks generated after the last pruning step.
            cur_masks: Dict{"layer_name": Tensor} that stores the current masks.
            scores: A dict{"layer_name": Tensor} that stores the pruning scores of weights.
            progressive_step: An integer representing the number of current step in progressive pruning.
            progressive_configs: A dict that stores configurations of progressive pruning.

        Returns:
            A dict{"layer_name": Tensor} that stores the masks generated in progressive pruning.
        """
        # three types: score-global (nxm and n:m), score-local(nxm and n:m), linear (only nxm)
        # score-local is a special type of score global therefore can be implemented with only one function
        progressive_steps = progressive_configs["progressive_steps"]
        progressive_masks = {}
        global_new_added_score_list = []
        new_added_masks = ProgressivePatternUtils.update_new_added_masks(pre_masks, cur_masks)
        new_added_masks_cnts = ProgressivePatternUtils.count_new_masked_cnts(new_added_masks)
        kth_masked_position = (new_added_masks_cnts * progressive_step) // progressive_steps
        for key in scores.keys():
            # block_size = self.block_size[key]
            # mask_num_each_block = progressive_step * int((block_size[0] * block_size[1]) // progressive_steps)
            new_added_filter = 1 - new_added_masks[key]
            new_added_cnts = torch.nonzero(new_added_filter).size()[0]
            score = scores[key]

            score_masked = (score * new_added_filter).abs()
            score_masked_row, _ = torch.sort(score_masked.flatten(), descending=True)
            score_masked_row = score_masked_row[:new_added_cnts]
            global_new_added_score_list.append(score_masked_row)

        global_new_added_scores = torch.cat(global_new_added_score_list, dim=0)
        if global_new_added_scores.size()[0] == 0:
            # an empty tensor, at target sparsity is 0 situation
            return pre_masks
        threshold, _ = torch.kthvalue(global_new_added_scores, kth_masked_position, dim=0)
        for key in scores.keys():  # pragma: no cover
            new_added_mask = new_added_masks[key]
            score = scores[key]
            new_added_filter = 1 - new_added_mask
            score_masked = (score * new_added_filter).abs()
            zero = torch.tensor([0.0]).to(score.device)
            one = torch.tensor([1.0]).to(score.device)
            progressive_mask = (new_added_mask + torch.where(score_masked <= threshold, zero, one)) * pre_masks[key]
            progressive_masks[key] = progressive_mask
        return progressive_masks

    @staticmethod
    def update_progressive_masks_local_scores(
        pre_masks, cur_masks, scores, progressive_step, progressive_configs
    ):  # pragma: no cover
        """Generate the progressive masks.

        Args:
            pre_masks: Dict{"layer_name": Tensor} that stores the masks generated after the last pruning step.
            cur_masks: Dict{"layer_name": Tensor} that stores the current masks.
            scores: A dict{"layer_name": Tensor} that stores the pruning scores of weights.
            progressive_step: An integer representing the number of current step in progressive pruning.
            progressive_configs: A dict that stores configurations of progressive pruning.

        Returns:
            A dict{"layer_name": Tensor} that stores the masks generated in progressive pruning.
        """
        # local is a speicial type of global, therefore we can call global to implement this
        progressive_steps = progressive_configs["progressive_steps"]
        progressive_masks = {}
        for key in scores.keys():
            # for local use
            pre_masks_for_this = {key: pre_masks[key]}
            cur_masks_for_this = {key: cur_masks[key]}
            scores_for_this = {key: scores[key]}
            progressive_masks_for_this = ProgressivePatternUtils.update_progressive_masks_global_scores(
                pre_masks_for_this, cur_masks_for_this, scores_for_this, progressive_step, progressive_configs
            )
            progressive_masks.update(progressive_masks_for_this)
        return progressive_masks

    @staticmethod
    def update_progressive_masks_scores_order(pre_masks, cur_masks, scores, progressive_step, progressive_configs):
        """Generate the progressive masks.

        Args:
            pre_masks: Dict{"layer_name": Tensor} that stores the masks generated after the last pruning step.
            cur_masks: Dict{"layer_name": Tensor} that stores the current masks.
            scores: A dict{"layer_name": Tensor} that stores the pruning scores of weights.
            progressive_step: An integer representing the number of current step in progressive pruning.
            progressive_configs: A dict that stores configurations of progressive pruning.

        Returns:
            A dict{"layer_name": Tensor} that stores the masks generated in progressive pruning.
        """
        if progressive_configs["use_global"]:
            return ProgressivePatternUtils.update_progressive_masks_global_scores(
                pre_masks, cur_masks, scores, progressive_step, progressive_configs
            )
        else:
            return ProgressivePatternUtils.update_progressive_masks_local_scores(
                pre_masks, cur_masks, scores, progressive_step, progressive_configs
            )

    @staticmethod
    def update_progressive_masks_linear_order(
        pre_masks, cur_masks, scores, progressive_step, progressive_configs: dict, block_sizes: dict
    ):
        """Generate the progressive masks.

        Args:
            pre_masks: Dict{"layer_name": Tensor} that stores the masks generated after the last pruning step.
            cur_masks: Dict{"layer_name": Tensor} that stores the current masks.
            scores: A dict{"layer_name": Tensor} that stores the pruning scores of weights.
            progressive_step: An integer representing the number of current step in progressive pruning.
            progressive_configs: A dict that stores configurations of progressive pruning.
            block_size: Dict{"layer_name": List or Tuple} that stores the block sizes, only for NxM patterns.

        Returns:
            A dict{"layer_name": Tensor} that stores the masks generated in progressive pruning.
        """
        progressive_steps = progressive_configs["progressive_steps"]
        progressive_masks = {}
        new_added_masks = ProgressivePatternUtils.update_new_added_masks(pre_masks, cur_masks)
        for key in pre_masks.keys():
            block_size = block_sizes[key]
            new_added_mask = new_added_masks[key]
            # conv
            new_added_mask = ProgressivePatternUtils._reshape_orig_to_2dims(new_added_mask)
            shape = new_added_mask.shape
            # progressive masks are generated in the direction of block's large dim.
            if block_size[0] >= block_size[1]:
                # NxM (N>=M), output channel pruning
                new_shape = [
                    shape[0] // block_size[0],
                    progressive_steps,
                    block_size[0] // progressive_steps,
                    shape[1] // block_size[1],
                    block_size[1],
                ]
                new_added_mask_reshape = new_added_mask.reshape(new_shape)
                new_added_mask_reshape[:, progressive_step:, :, :, :] = 1.0
            else:
                # NxM (N<M), input channel pruning
                new_shape = [
                    shape[0] // block_size[0],
                    block_size[0],
                    shape[1] // block_size[1],
                    progressive_steps,
                    block_size[1] // progressive_steps,
                ]
                new_added_mask_reshape = new_added_mask.reshape(new_shape)
                new_added_mask_reshape[:, :, :, progressive_step:, :] = 1.0
            new_added_mask = new_added_mask_reshape.reshape(shape)
            new_added_mask = ProgressivePatternUtils._reshape_2dims_to_orig(new_added_mask, pre_masks[key].shape)
            progressive_masks[key] = pre_masks[key] * new_added_mask
        return progressive_masks


class BasePattern:
    """Pruning Pattern.

    It defines the basic pruning unit and how this unit will be pruned during pruning, e.g. 4x1, 2:4.

    Args:
        config: A config dict object that contains the pattern information.
        modules: neural network modules to be pruned with the pattern.

    Attributes:
        pattern: A config dict object that includes information of the pattern.
        is_global:  A bool determining whether the pruning takes global pruning option.
                    Global pruning means that pruning scores by a pruning criterion are evaluated in all layers.
                    Local pruning, by contrast, means that pruning scores by the pruning criterion are evaluated
                        in every layer individually.
        keep_mask_layers:A dict that includes the layers whose mask will not be updated.
        invalid_layers: The layers whose shapes don't fit the pattern.
        modules: neural network modules to be pruned with the pattern.
        config: A config dict object that contains all the information including the pattern's.
        max_sparsity_ratio_per_op: A float representing the maximum sparsity that one layer could reach.
        min_sparsity_ratio_per_op: A float representing the minimum sparsity that one layer could reach.
        target_sparsity: A float representing the sparsity ratio of the modules after pruning.
    """

    def __init__(self, config, modules):
        """Initialize the basic pruning unit of a pattern."""
        self.pattern = config.pattern
        self.is_global = config.pruning_scope == "global"
        self.keep_mask_layers = {}
        self.invalid_layers = []
        self.modules = modules
        self.config = config
        self.max_sparsity_ratio_per_op = self.config["max_sparsity_ratio_per_op"]
        self.min_sparsity_ratio_per_op = self.config["min_sparsity_ratio_per_op"]
        self.target_sparsity_ratio = self.config["target_sparsity"]
        self.block = bool("block" in self.config["pruning_type"] or "retrain" in self.config["pruning_type"])
        # Not using deterministic_algorithms for all examples

    def get_masks(self, scores, target_sparsity_ratio, pre_masks):
        """Generate the weight masks according to the weight score and the current target sparsity ratio.

        Args:
            scores: A dict{"layer_name": Tensor} that stores the pruning scores of weights.
            target_sparsity_ratio: A float representing the sparsity of the modules after pruning.
            pre_masks: A dict{"layer_name": Tensor} that stores the masks generated at last pruning step.

        Returns:
            A dict with the identical size as pre_masks and its 0/1 values are updated.
                1 means unpruned and 0 means pruned.
        """
        if self.is_global:
            return self.get_masks_global(scores, target_sparsity_ratio, pre_masks)
        else:
            return self.get_masks_local(scores, target_sparsity_ratio, pre_masks)

    def get_masks_global(self, scores, target_sparsity_ratio, pre_masks):
        """Generate the weight masks for global pruning, please refer to function get_masks for more information."""
        raise NotImplementedError

    def get_masks_local(self, scores, target_sparsity_ratio, pre_masks):
        """Generate the weight masks for local pruning.

        Args:
            scores: A dict{"layer_name": Tensor} that stores the pruning scores of weights.
            target_sparsity_ratio: A float. After pruning, the sparsity of the modules will reach this value.
            pre_masks: A dict{"layer_name": Tensor}. The previous masks generated at the last pruning step.

        Returns:
            A dict with the identical size as pre_masks and its 0/1 values are updated.
                1 means unpruned and 0 means pruned.
        """
        masks = {}
        for key in scores.keys():
            score = {key: scores[key]}
            pre_mask = {key: pre_masks[key]}
            mask = self.get_masks_global(score, target_sparsity_ratio, pre_mask)
            masks[key] = mask[key]
        return masks

    def get_block_size_dict(self, data):
        """Get pattern size for each module.

        This is mainly for per-channel pruning when each module has different pruning size.

        Args:
            data: the input data.

        Returns:
            To be implemented in subclasses.
        """
        raise NotImplementedError

    def get_sparsity_ratio(self, pre_masks, return_dict=False):
        """Calculate the zero elements' ratio in pre_masks.

        Args:
            pre_masks: Dict{"layer_name": Tensor} that stores the masks generated after the last pruning step.
            return_dict: A bool determining whether to return more information like zero_cnt and total_cnt.

        Returns:
            A float representing the zero elements' ratio in pre_masks.
        """
        raise NotImplementedError

    def check_layer_validity(self):
        """Check if a layer is valid for this block_size."""
        pass

    def get_reduced_masks_from_data(self, data, key):
        """Obtain the unpruned weights and reshape according to the block_size."""
        raise NotImplementedError

    def get_sparsity_ratio_each_layer(self, mask):
        """Calculate the sparsity ratio of each layer."""
        raise NotImplementedError

    def update_residual_cnt(self, masks, target_sparsity_ratio):
        """Update the number of parameters yet to be pruned.

        Args:
            masks: the current pruning mask.
            target_sparsity_ratio: A float representing the final sparsity of the modules.

        Returns:
            An int representing the number of weights left to be pruned to reach the target sparsity ratio.
        """
        self.total_params_cnt = self.get_sparsity_ratio(masks, return_dict=True)["total_cnt"]
        to_prune_cnt = int(self.total_params_cnt * target_sparsity_ratio)
        for key in masks.keys():
            if self.keep_mask_layers.get(key, False):
                zero_cnt = self.get_sparsity_ratio({key: masks[key]}, return_dict=True)["zero_cnt"]
                to_prune_cnt -= zero_cnt

        return to_prune_cnt

    def adjust_ratio(
        self,
        masks: dict,
        layer_name: str,
        key_new_sparsity: SparsityInfo,
        max_sparsity_ratio: float,
        min_sparsity_ratio: float,
        final_target_sparsity_ratio: float,
    ):
        """Adjust the sparsity of a layer based on threshold.

        Args:
            masks: The weight masks.
            layer_name: The layer to be examined.
            key_new_sparsity: The proposed ratio for the layer.
            max_sparsity_ratio: A float representing the maximum sparsity that one layer could reach.
            min_sparsity_ratio: A float representing the minimum sparsity that one layer could reach.
            final_target_sparsity_ratio: The final target sparsity ratio.

        Returns:
            A bool indicating if the ratio needs to be adjusted.
            adjust_sparsity_ratio: The adjusted sparsity ratio.
        """
        need_adjust = False
        adjust_zero_cnt = key_new_sparsity.zero_cnt
        adjust_sparsity_ratio = key_new_sparsity.sparsity_ratio
        adjust_total_cnt = key_new_sparsity.total_cnt

        if adjust_sparsity_ratio > max_sparsity_ratio:
            need_adjust = True
            adjust_sparsity_ratio = max_sparsity_ratio
            adjust_zero_cnt = int(adjust_total_cnt * max_sparsity_ratio)

        if adjust_sparsity_ratio < min_sparsity_ratio:
            return need_adjust, adjust_sparsity_ratio

        # TODO no need to calculate each time
        infos, net_info = self.get_sparsity_ratio_each_layer(masks)

        any_exceed_target_ratio = False
        for key in infos.keys():
            if infos[key].sparsity_ratio > final_target_sparsity_ratio:
                any_exceed_target_ratio = True
                break
        if adjust_sparsity_ratio > final_target_sparsity_ratio:
            any_exceed_target_ratio = True
        if not any_exceed_target_ratio:
            return need_adjust, adjust_sparsity_ratio

        zero_cnt_below_min_sparsity = 0
        total_cnt_below_min_sparsity = 0
        zero_cnt_above_min_sparsity = 0
        for key in infos.keys():
            info = infos[key]
            if key == layer_name:
                info = SparsityInfo(
                    zero_cnt=adjust_zero_cnt, total_cnt=adjust_total_cnt, sparsity_ratio=adjust_sparsity_ratio
                )
            if info.sparsity_ratio < min_sparsity_ratio:
                zero_cnt_below_min_sparsity += info.zero_cnt
                total_cnt_below_min_sparsity += info.total_cnt
            else:
                zero_cnt_above_min_sparsity += info.zero_cnt

        gap_cnt = int(total_cnt_below_min_sparsity * min_sparsity_ratio) - zero_cnt_below_min_sparsity
        remaining_cnt = (
            int(net_info.total_cnt * final_target_sparsity_ratio)
            - zero_cnt_above_min_sparsity
            - zero_cnt_below_min_sparsity
        )
        if remaining_cnt >= gap_cnt:
            return need_adjust, adjust_sparsity_ratio
        else:
            new_zero_cnt = adjust_zero_cnt - (gap_cnt - remaining_cnt)
            new_sparsity_ratio = float(new_zero_cnt) / adjust_total_cnt
            # adjust_zero_cnt = new_zero_cnt
            adjust_sparsity_ratio = new_sparsity_ratio
            return True, adjust_sparsity_ratio


class PytorchBasePattern(BasePattern):
    def __init__(self, config, modules):
        super().__init__(config, modules)
        #   If you need to use it, you can set it in example
        #   and start the environment variable: exaport CUBLAS_WORKSPACE_CONFIG=:'4096:8'
        # torch.use_deterministic_algorithms(True, warn_only=True)

    def reduce_tensor(self, data, dim):
        """Reduce the data along the given dimension.

        Args:
            data: The input data.
            dim: The reduced axis.

        Returns:
            The reduced tensor.
        """
        name = self.config["criterion_reduce_type"]
        if name == "mean":
            return torch.mean(data, dim=dim)
        elif name == "sum":
            return torch.sum(data, dim=dim)
        elif name == "max":
            return torch.max(data, dim=dim)[0]
        else:
            assert False, "currently only support mean, sum and max reduce type"

    def get_single_mask_per_target_ratio(self, score, exact_sparsity_ratio):
        """Generate a mask for one layer with the exact_sparsity_ratio.

        Args:
            score: A Tensor representing the pruning scores of each weight elements.
            exact_sparsity_ratio: A float representing the layer's final sparsity ratio.

        Returns:
            A Tensor with the identical size as score. a new mask.
        """
        flattern_score = torch.flatten(score)
        k = int(exact_sparsity_ratio * flattern_score.numel())
        if not k < 1:
            threshold, _ = torch.kthvalue(flattern_score, k)
            zero = torch.tensor([False]).to(score.device)
            one = torch.tensor([True]).to(score.device)
            mask = torch.where(score <= threshold, zero, one)
        else:  # pragma: no cover
            mask = torch.ones(score.shape, device=score.device)
        if self.block:
            mask = mask.float()
        return mask

    def get_sparsity_ratio(self, pre_masks, return_dict=False):
        """Calculate the zero elements' ratio in pre_masks.

        Args:
            pre_masks: Dict{"layer_name": Tensor} that stores the masks generated after the last pruning step.
            return_dict: A bool determining whether to return more information like zero_cnt and total_cnt.

        Returns:
            A float representing the zero elements' ratio in pre_masks.
        """
        zero_cnt = 0  # This function might need to refactor in subclass.
        total_cnt = 0
        for key in pre_masks.keys():
            pre_mask = pre_masks[key]
            zero_cnt += torch.sum(pre_mask == 0.0).data.item()
            total_cnt += pre_mask.numel()  # FIXME

        if return_dict:
            return {"sparsity_ratio": float(zero_cnt) / total_cnt, "zero_cnt": zero_cnt, "total_cnt": total_cnt}
        else:
            return float(zero_cnt) / total_cnt

    def get_sparsity_ratio_progressive(self, pre_masks, return_dict=False):
        """Calculate the sparsity ratio of each layer.

        Args:
            pre_masks: Dict{"layer_name": Tensor} that stores the masks generated after the last pruning step.
            return_dict: A bool determining whether to return more information like zero_cnt and total_cnt.

        Returns:
            A float representing the zero elements' ratio in pre_masks.
        """
        zero_cnt = 0
        total_cnt = 0
        for key in pre_masks.keys():
            if key in self.invalid_layers:
                continue
            # progressive masks are unstructured, therefore directly find zeros
            zero_cnt += float(torch.sum(pre_masks[key] == 0.0).data.item())
            total_cnt += float(pre_masks[key].numel())

        return zero_cnt / total_cnt

    def get_pattern_lock_masks(self, modules):
        """Obtain masks from original weight map according the pattern and weights' zero positions.

        Args:
            modules: a dict {'layer_name': Tensor} that stores weights.

        Returns:
            A dict with the identical size as modules, containing pattern lock masks.
        """
        pattern_lock_masks = {}
        for key in modules.keys():
            # weight = modules[key].weight
            # shape = weight.shape
            param = modules[key].weight
            data = safe_get_data(param)
            shape = safe_get_shape(param)
            mask = torch.ones(shape)
            # mask[weight == 0] = 0.0
            mask[data == 0] = 0.0
            mask = mask.bool()
            pattern_lock_masks[key] = mask.to(param.device)

        return pattern_lock_masks

    def get_sparsity_ratio_each_layer(self, masks):
        """Calculate the sparsity ratio of each layer.

        Args:
            masks: The current weight masks.

        Returns:
            infos: the sparsity information for each layer including sparsity_ratio, zero_point and total cnts.
            SparsityInfo: the sparsity information for the model.
        """
        infos = {}
        zero_cnts = 0
        total_cnts = 0

        for key in masks.keys():
            if key in self.invalid_layers:
                continue
            reduced_mask = masks[key] if self.block else self.get_reduced_masks_from_data(masks[key], key)

            zero_cnt = int(torch.sum(reduced_mask == 0.0).data.item())
            total_cnt = int(reduced_mask.numel())
            sparsity_ratio = float(zero_cnt) / total_cnt
            val = SparsityInfo(zero_cnt, total_cnt, sparsity_ratio)
            infos[key] = val
            zero_cnts += zero_cnt
            total_cnts += total_cnt

        sparsity_ratio = float(zero_cnts) / total_cnts
        return infos, SparsityInfo(zero_cnts, total_cnts, sparsity_ratio)


class KerasBasePattern(BasePattern):
    def __init__(self, config, modules):
        super().__init__(config, modules)

    def reduce_tensor(self, data, dim):
        """Reduce the data along the given dimension.

        Args:
            data: The input data.
            dim: The reduced axis.

        Returns:
            The reduced tensor.
        """
        name = self.config["criterion_reduce_type"]
        if name == "mean":
            return tf.math.reduce_mean(data, dim)
        elif name == "sum":
            return tf.math.reduce_sum(data, dim)
        elif name == "max":
            return tf.math.reduce_max(data, dim)
        else:
            assert False, "currently only support mean, sum and max reduce type"

    def get_single_mask_per_target_ratio(self, score, exact_sparsity_ratio):
        """Generate a mask for one layer with the exact_sparsity_ratio.

        Args:
            score: A Tensor representing the pruning scores of each weight elements.
            exact_sparsity_ratio: A float representing the layer's final sparsity ratio.

        Returns:
            A Tensor with the identical size as score. a new mask.
        """
        flattern_score = tf.reshape(score, [-1]).numpy()
        k = int(exact_sparsity_ratio * flattern_score.size)
        threshold = np.partition(flattern_score, kth=k)[k]
        if not k < 1:
            zero = tf.convert_to_tensor([0.0])
            one = tf.convert_to_tensor([1.0])
            mask = tf.where(score <= threshold, zero, one)
        else:
            mask = tf.ones_like(score)
        return mask

    def get_sparsity_ratio_each_layer(self, masks):
        """Calculate the sparsity ratio of each layer.

        Args:
            masks: The current weight masks.

        Returns:
            infos: the sparsity information for each layer including sparsity_ratio, zero_point and total cnts.
            SparsityInfo: the sparsity information for the model.
        """
        infos = {}
        zero_cnts = 0
        total_cnts = 0

        for key in masks.keys():
            if key in self.invalid_layers:
                continue
            if not isinstance(masks[key], np.ndarray):
                masks[key] = masks[key].numpy()
            reduced_mask = masks[key] if self.block else self.get_reduced_masks_from_data(masks[key], key)
            zero_cnt = int(np.sum(reduced_mask == 0.0))
            total_cnt = int(reduced_mask.size)
            sparsity_ratio = float(zero_cnt) / total_cnt
            val = SparsityInfo(zero_cnt, total_cnt, sparsity_ratio)
            infos[key] = val
            zero_cnts += zero_cnt
            total_cnts += total_cnt

        sparsity_ratio = float(zero_cnts) / total_cnts
        return infos, SparsityInfo(zero_cnts, total_cnts, sparsity_ratio)
