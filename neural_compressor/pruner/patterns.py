"""pruning patterns."""
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

import logging

from neural_compressor.utils.utility import LazyImport
torch = LazyImport('torch')
from .logger import logger
from collections import namedtuple

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


def get_pattern(config, modules):
    """Get registered pattern class.

    Get a Pattern object from PATTERNS.

    Args:
        config: A config dict object that contains the pattern information.
        modules: Torch neural network modules to be pruned with the pattern.

    Returns:
        A Pattern object.

    Raises:
        AssertionError: Currently only support patterns which have been registered in PATTERNS.
    """
    name = config.pattern
    name = name.split('_')[-1]
    if "x" in name:
        return PATTERNS["NxM"](config, modules)
    if ":" in name:
        return PATTERNS["N:M"](config, modules)
    assert False, f"currently only support {PATTERNS.keys()}"


SparsityInfo = namedtuple("SparsityInfo", ['zero_cnt', 'total_cnt', 'sparsity_ratio'])


class BasePattern:
    """Pruning Pattern.

    It defines the basic pruning unit and how this unit will be pruned during pruning, e.g. 4x1, 2:4.
    
    Args:
        config: A config dict object that contains the pattern information.
        modules: Torch neural network modules to be pruned with the pattern.

    Attributes:
        pattern: A config dict object that includes information of the pattern.    
        is_global:  A bool determining whether the pruning takes global pruning option.
                    Global pruning means that pruning scores by a pruning criterion are evaluated in all layers.
                    Local pruning, by contrast, means that pruning scores by the pruning criterion are evaluated 
                        in every layer individually.
        keep_mask_layers:A dict that includes the layers whose mask will not be updated.
        invalid_layers: The layers whose shapes don't fit the pattern.
        modules: Torch neural network modules to be pruned with the pattern.
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
        self.max_sparsity_ratio_per_op = self.config['max_sparsity_ratio_per_op']
        self.min_sparsity_ratio_per_op = self.config['min_sparsity_ratio_per_op']
        self.target_sparsity_ratio = self.config['target_sparsity']
        # Not using deterministic_algorithms for all examples
        torch.use_deterministic_algorithms(False)

    def reduce_tensor(self, data, dim):
        """Reduce the data along the given dimension.
        
        Args:
            data: The input data.
            dim: The reduced axis.

        Returns:
            The reduced tensor.
        """
        name = self.config['criterion_reduce_type']
        if name == "mean":
            return torch.mean(data, dim=dim)
        elif name == "sum":
            return torch.sum(data, dim=dim)
        elif name == "max":
            return torch.max(data, dim=dim)[0]
        else:
            assert False, "currently only support mean, sum and max reduce type"

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
        if isinstance(self, PatternNxM) and not isinstance(self.block_size, dict):
            self.block_size = self.get_block_size_dict(pre_masks)
        for key in scores.keys():
            score = {key: scores[key]}
            pre_mask = {key: pre_masks[key]}
            mask = self.get_masks_global(score, target_sparsity_ratio, pre_mask)
            masks[key] = mask[key]
        return masks

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
        threshold, _ = torch.kthvalue(flattern_score, k)
        if not k < 1:
            zero = torch.tensor([0.]).to(score.device)
            one = torch.tensor([1.]).to(score.device)
            mask = torch.where(score <= threshold, zero, one)
        else:
            mask = torch.ones(score.shape, device=score.device)
        return mask

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
        zero_cnt = 0 # This function might need to refactor in subclass.
        total_cnt = 0
        for key in pre_masks.keys():
            pre_mask = pre_masks[key]
            zero_cnt += torch.sum(pre_mask == 0.0).data.item()
            total_cnt += pre_masks[key].numel()  ##FIXME
        if return_dict:
            return {"sparsity_ratio": float(zero_cnt) / total_cnt, "zero_cnt": zero_cnt, "total_cnt": total_cnt}
        else:
            return float(zero_cnt) / total_cnt

    def get_pattern_lock_masks(self, modules):
        """Obtain masks from original weight map according the pattern and weights' zero positions.

        Args:
            modules: a dict {'layer_name': Tensor} that stores weights.

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

    def check_layer_validity(self):
        """Check if a layer is valid for this block_size."""
        pass

    def get_reduced_masks_from_data(self, data, key):
        """Obtain the unpruned weights and reshape according to the block_size."""
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
            reduced_mask = self.get_reduced_masks_from_data(masks[key], key)
            zero_cnt = (int(torch.sum(reduced_mask == 0.0).data.item()))
            total_cnt = int(reduced_mask.numel())
            sparsity_ratio = float(zero_cnt) / total_cnt
            val = SparsityInfo(zero_cnt, total_cnt, sparsity_ratio)
            infos[key] = val
            zero_cnts += zero_cnt
            total_cnts += total_cnt
        sparsity_ratio = float(zero_cnts) / total_cnts
        return infos, SparsityInfo(zero_cnts, total_cnts, sparsity_ratio)

    def adjust_ratio(self, masks: dict, layer_name: str, key_new_sparsity: SparsityInfo,
                     max_sparsity_ratio: float, min_sparsity_ratio: float, \
                     final_target_sparsity_ratio: float):
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

        ##TODO no need to calculate each time
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
                info = SparsityInfo(zero_cnt=adjust_zero_cnt, total_cnt=adjust_total_cnt,
                                    sparsity_ratio=adjust_sparsity_ratio)
            if info.sparsity_ratio < min_sparsity_ratio:
                zero_cnt_below_min_sparsity += info.zero_cnt
                total_cnt_below_min_sparsity += info.total_cnt
            else:
                zero_cnt_above_min_sparsity += info.zero_cnt

        gap_cnt = int(total_cnt_below_min_sparsity * min_sparsity_ratio) - zero_cnt_below_min_sparsity
        remaining_cnt = int(net_info.total_cnt * final_target_sparsity_ratio) \
                        - zero_cnt_above_min_sparsity - zero_cnt_below_min_sparsity
        if remaining_cnt >= gap_cnt:
            return need_adjust, adjust_sparsity_ratio
        else:
            new_zero_cnt = adjust_zero_cnt - (gap_cnt - remaining_cnt)
            new_sparsity_ratio = float(new_zero_cnt) / adjust_total_cnt
            ##adjust_zero_cnt = new_zero_cnt
            adjust_sparsity_ratio = new_sparsity_ratio
            return True, adjust_sparsity_ratio


@register_pattern('NxM')
class PatternNxM(BasePattern):
    """Pruning Pattern.
    
    A Pattern class derived from BasePattern. In this pattern, the weights in a NxM block will be pruned or kept
    during one pruning step.
    
    Args:
        config: A config dict object that contains the pattern information.
        
    Attributes:
            block_size: A list of two integers representing the height and width of the block.
            Please note that the vertical direction of a Linear layer's weight refers to the output channel.
                because PyTorch's tensor matmul has a hidden transpose operation.
    """

    def __init__(self, config, modules):
        """Initialize the basic pruning unit of NXM pattern."""
        super(PatternNxM, self).__init__(config, modules)
        pattern = self.pattern.split('_')[-1]
        self.N = pattern.split('x')[0]
        self.M = pattern.split('x')[1]
        if self.N == "channel":  ##channel-wise pruning mode
            self.block_size = ["channel", int(self.M)]
        elif self.M == "channel":  ##channel-wise pruning mode
            self.block_size = [int(self.N), "channel"]
        else:
            self.block_size = [int(pattern.split('x')[0]), int(pattern.split('x')[1])]
        self.total_params_cnt = -1

        self.block_size = self.get_block_size_dict()
        self.check_layer_validity()

    def get_block_size_dict(self):
        """Calulate the zero elements' ration in pre_masks.
        
        Args:
            data: Dict{"layer_name": Tensor} that stores weights or scores.
            
        Returns:
            A dict. Dict{"layer_name": [block_size_1, block_size_2]} containing block shapes of each layer.
                In channel-wise pruning different layers can have different pruning patterns.
        """
        data = self.modules
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

    def check_layer_validity(self):
        """Check if a layer is valid for this block_size."""
        block_sizes = self.block_size
        datas = self.modules
        for key in datas.keys():
            data = datas[key].weight
            data = self._reshape_orig_to_2dims(data)
            shape = data.shape
            block_size = block_sizes[key]
            if shape[0] % block_size[0] != 0 or shape[1] % block_size[1] != 0:  ## only consider input channel
                self.invalid_layers.append(key)
                logger.warning(f"{key} shape {data.shape} cannot be divided by {self.pattern}")

    def get_reduced_masks_from_data(self, data, key):
        """Obtain the unpruned weights and reshape according to the block_size.
        
        Args:
            data: Input.
            key: The layer name.
        
        Returns:
            The unpruned weights.
        """
        assert key not in self.invalid_layers
        block_size = self.block_size[key]
        data = self._reshape_orig_to_2dims(data)
        shape = data.shape
        new_shape = [shape[0] // block_size[0], block_size[0], shape[1] // block_size[1], block_size[1]]
        data = data.reshape(new_shape)
        data = data.sum(-1).sum(1)
        reduced_mask = data != 0
        return reduced_mask

    def get_sparsity_ratio(self, pre_masks, return_dict=False):
        """Please note that the zero cnt and total cnt are all block_wise for supporting channel-wise pruning.

        Args:
            pre_masks: Dict{"layer_name": Tensor} representing the masks generated after the last pruning step.
            return_dict: A bool determining whether to return more information like zero_cnt and total_cnt.

        Returns:
            A float representing the zero elements' ratio in pre_masks.
        """
        zero_cnt = 0
        total_cnt = 0
        for key in pre_masks.keys():
            if key in self.invalid_layers:
                continue
            reduced_mask = self.get_reduced_masks_from_data(pre_masks[key], key)
            zero_cnt += (int(torch.sum(reduced_mask == 0.0).data.item()))
            total_cnt += int(reduced_mask.numel())
        if total_cnt == 0:
            sparsity_ratio = 0.0
        else:
            sparsity_ratio = float(zero_cnt) / total_cnt
        if return_dict:
            return {"sparsity_ratio": sparsity_ratio, "zero_cnt": zero_cnt, "total_cnt": total_cnt}
        else:
            return sparsity_ratio

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
            zero_cnt += float(torch.sum(pre_masks[key] == 0).data.item())
            total_cnt += float(pre_masks[key].numel())
        return (zero_cnt / total_cnt)

    def _reshape_orig_to_2dims(self, data):
        """Process layers that are not two-dimensional(e.g conv layer).

        Args:
            data: Input.
            
        Returns:
            Reshaped data.
        """
        ##TODO need to verify whether it's ok for transposed conv
        if len(data.shape) == 4:
            data = data.permute(0, 2, 3, 1)  ##cout,k,k,cin
            data = data.reshape(data.shape[0], -1)
        return data

    def _reshape_2dims_to_orig(self, data, orig_shape):
        """Recover layers that are not two-dimensional(e.g conv layer).

        Args:
            data: Input.
            orig_shape: Target shape.
            
        Returns:
            Reshaped data.
        """
        if len(orig_shape) == 4:
            data = data.reshape(orig_shape[0], orig_shape[2], orig_shape[3],
                                orig_shape[1])
            data = data.permute(0, 3, 1, 2)
        return data

    def reshape_orig_to_pattern(self, data, key):
        """Reshape the data(s1,s2) to [s1/N,N,s2,s2/M].

        Args:
            data: The input.
            key: The layer name.
            
        Returns:
            Reshaped input tensor.
        """
        block_size = self.block_size[key]
        data = self._reshape_orig_to_2dims(data)
        shape = data.shape
        new_shape = [shape[0] // block_size[0], block_size[0], shape[1] // block_size[1],
                     block_size[1]]
        data = data.reshape(new_shape)
        return data

    def reshape_reduced_to_orig(self, data, key, orig_shape):
        """Reshape the data [s1/N,s2/M] to [s1,s2], also permute dims for conv layer.

        Args:
            data: Input.
            key: The layer name.
            orig_shape: The original shape of the layer.
            
        Returns:
            Data of its original shape.
        """
        block_size = self.block_size[key]
        data = data.repeat_interleave(block_size[0], dim=0).repeat_interleave(block_size[1], dim=-1)
        data = self._reshape_2dims_to_orig(data, orig_shape)
        return data

    def reduce_scores(self, scores):
        """Recalculate the pruning scores after reducing the data.
        
        Args:
            scores: A dict{"layer_name": Tensor} that stores the pruning scores of weights.

        Returns:
            The reduced pruning scores.
        """
        new_scores = {}
        for key in scores.keys():
            if key in self.invalid_layers:
                continue
            if self.keep_mask_layers.get(key, False):
                continue
            self.keep_mask_layers[key] = False
            current_score = scores[key]
            current_score = self.reshape_orig_to_pattern(current_score, key)
            ##sum or mean is quite different for per channel pruning
            current_score_sum = self.reduce_tensor(self.reduce_tensor(current_score, dim=-1), dim=1)
            new_scores[key] = current_score_sum
        return new_scores

    def get_mask_per_threshold(self, score, threshold, block_size):
        """Get the mask per threshold."""
        zero = torch.tensor([0.]).to(score.device)
        one = torch.tensor([1.]).to(score.device)
        mask = torch.where(score <= threshold, zero, one)
        mask = mask.repeat_interleave(block_size[0], dim=0).repeat_interleave(block_size[1], dim=-1)
        return mask

    def get_masks_global(self, scores, cur_target_sparsity_ratio, pre_masks,
                         keep_exact_sparsity_ratio=True):
        """Generate masks for layers.

        Gather all layer's scores together and calculate a common threshold.
        This threshold will be applied to all layers.
        
        Args:
            scores: A dict{"layer_name": Tensor} that stores the pruning scores of weights.
            cur_target_sparsity_ratio: A float representing the model's sparsity after pruning.
            pre_masks: A dict{"layer_name": Tensor} that stores the masks generated at the last pruning step.
            max_sparsity_ratio_per_op: A float representing the maximum sparsity that one layer can reach.
            keep_pre_masks: A bool representing if the masks should remain unchanged.
            
        Returns:
            A dict with the identical size as pre_masks and its 0/1 values are updated. 
                1 means unpruned and 0 means pruned.
        """
        ##keep the masks if the layer exceed max sparsity ratio

        masks = pre_masks

        k_blockwise = self.update_residual_cnt(masks, cur_target_sparsity_ratio)
        if k_blockwise <= 0:
            return masks
        new_scores = self.reduce_scores(scores)
        global_scores = torch.cat([torch.flatten(v) for v in new_scores.values()])
        residual_k = k_blockwise
        not_exceed_layers = [key for key in new_scores.keys()]
        if self.min_sparsity_ratio_per_op > 0:
            sparsity_infos_perlayer, _ = self.get_sparsity_ratio_each_layer(masks)

        while True:
            threshold, _ = torch.kthvalue(global_scores, residual_k)
            for key in not_exceed_layers:
                block_size = self.block_size[key]
                score = new_scores[key]
                mask = self.get_mask_per_threshold(score, threshold, block_size)
                info = self.get_sparsity_ratio({key: mask}, return_dict=True)
                zero_cnt = info["zero_cnt"]
                total_cnt = info["total_cnt"]
                current_sparsity_ratio = float(zero_cnt) / total_cnt
                key_new_sparsity = SparsityInfo(zero_cnt, total_cnt, current_sparsity_ratio)
                need_adjust, adjust_ratio = self.adjust_ratio(masks, key, key_new_sparsity,
                                                              self.max_sparsity_ratio_per_op,
                                                              self.min_sparsity_ratio_per_op,
                                                              self.target_sparsity_ratio)
                if need_adjust:
                    # uptade status
                    self.keep_mask_layers[key] = True
                    masks[key] = self.get_single_mask_per_target_ratio(new_scores[key], adjust_ratio)
                    masks[key] = masks[key].repeat_interleave(block_size[0], 0).repeat_interleave(block_size[1], -1)
                    if keep_exact_sparsity_ratio:
                        zero_cnt = self.get_sparsity_ratio({key: masks[key]}, return_dict=True)["zero_cnt"]
                        residual_k -= zero_cnt
                else:
                    masks[key] = mask
            if not keep_exact_sparsity_ratio:
                break
            new_not_exceed_layers = [key for key in new_scores.keys() if not self.keep_mask_layers.get(key, False)]
            if not_exceed_layers == new_not_exceed_layers or len(new_not_exceed_layers) == 0:
                break
            not_exceed_layers = new_not_exceed_layers
            global_scores = torch.cat([torch.flatten(new_scores[key]) for key in not_exceed_layers])

        for key in masks.keys():
            if key in self.invalid_layers:
                continue
            if len(scores[key].shape) == 4:  ## need to permute
                mask = masks[key]
                orig_shape = scores[key].shape
                mask = self._reshape_2dims_to_orig(mask, orig_shape)
                masks[key] = mask
            layer_ratio = torch.sum(masks[key] == 0.0).data.item() / masks[key].numel()
            logger.info(f'layer {key} sparsity_ratio is {layer_ratio}')
        return masks

    def get_pattern_lock_masks(self, modules):
        """Obtain masks from original weight map by masking the zero-valued weights.
        
        Args:
            modules: A dict{"layer_name": Tensor} that stores weights.
            
        Returns:
            A dict with the identical size as modules, containing pattern lock masks.
        """
        pattern_lock_masks = {}
        for key in modules.keys():
            weight = modules[key].weight
            ori_shape = weight.shape
            if key in self.invalid_layers:
                mask = torch.ones(weight.shape, device=weight.device)
                pattern_lock_masks[mask] = mask
                continue
            reduced_mask = self.get_reduced_masks_from_data(weight, key)
            mask = self.reshape_reduced_to_orig(reduced_mask, key, ori_shape)
            pattern_lock_masks[key] = mask
        return pattern_lock_masks

    # ---------------progressive related--------------------
    def count_new_masked_cnts(self, new_added_masks):
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

    def update_new_added_masks(self, pre_masks, cur_masks):
        """Obtain the new set-to-zero masks during a pruning procedure.

        Pre_masks, cur_masks should have identical keys bacause they represent the same model.

        Args:
            pre_masks: Dict{"layer_name": Tensor} that stores the masks generated after the last pruning step.
            cur_masks: Dict{"layer_name": Tensor} that stores the current masks.
        
        Returns:
            A dict {"layer_name": Tensor} that stores the added masks.
        """
        # obtain the new set-to-zero mask during a pruning procedure.
        # pre_masks, cur_masks should have identical keys bacause they stands for one model.
        new_added_masks = {}
        for key in pre_masks.keys():
            pre_mask = pre_masks[key]
            cur_mask = cur_masks[key]
            zero = torch.tensor([0.]).to(pre_mask.device)
            one = torch.tensor([1.]).to(cur_mask.device)
            new_added_masks[key] = torch.where(pre_mask == cur_mask, one, zero)
        return new_added_masks

    def update_progressive_masks(self, pre_masks, cur_masks, scores, progressive_step, progressive_configs):
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
        use_global = progressive_configs["use_global"]
        if use_global:
            return self.update_progressive_masks_global(pre_masks, cur_masks, scores, \
                                                        progressive_step, progressive_configs)
        else:
            return self.update_progressive_masks_local(pre_masks, cur_masks, scores, \
                                                       progressive_step, progressive_configs)

    def update_progressive_masks_linear(self, pre_masks, cur_masks, progressive_step, progressive_configs):
        """Generate the progressive masks along the block's larger dimension.
        
        Args:
            pre_masks: Dict{"layer_name": Tensor} that stores the masks generated after the last pruning step.
            cur_masks: Dict{"layer_name": Tensor} that stores the current masks.
            progressive_step: An integer representing the number of current step in progressive pruning.
            progressive_configs: A dict that stores configurations of progressive pruning.

        Returns:
            A dict{"layer_name": Tensor} that stores the masks generated in progressive pruning.
        """
        progressive_steps = progressive_configs["progressive_steps"]
        progressive_masks = {}
        new_added_masks = self.update_new_added_masks(pre_masks, cur_masks)
        for key in pre_masks.keys():
            block_size = self.block_size[key]
            new_added_mask = new_added_masks[key]
            # conv
            new_added_mask = self._reshape_orig_to_2dims(new_added_mask)
            shape = new_added_mask.shape
            # progressive masks are generated in the direction of block's large dim.
            if block_size[0] >= block_size[1]:
                # NxM (N>=M), output channel pruning
                new_shape = [shape[0] // block_size[0], progressive_steps, block_size[0] // progressive_steps,
                             shape[1] // block_size[1], block_size[1]]
                new_added_mask_reshape = new_added_mask.reshape(new_shape)
                new_added_mask_reshape[:, progressive_step:, :, :, :] = 1.0
            else:
                # NxM (N<M), input channel pruning
                new_shape = [shape[0] // block_size[0], block_size[0], shape[1] // block_size[1],
                             progressive_steps, block_size[1] // progressive_steps]
                new_added_mask_reshape = new_added_mask.reshape(new_shape)
                new_added_mask_reshape[:, :, :, progressive_step:, :] = 1.0
            new_added_mask = new_added_mask_reshape.reshape(shape)
            new_added_mask = self._reshape_2dims_to_orig(new_added_mask, pre_masks[key].shape)
            progressive_masks[key] = pre_masks[key] * new_added_mask
        return progressive_masks

    def update_progressive_masks_scores(self, pre_masks, cur_masks, scores, progressive_step, progressive_configs):
        """Generate the progressive masks based on scores.
        
        Args:
            pre_masks: Dict{"layer_name": Tensor} that stores the masks generated after the last pruning step.
            cur_masks: Dict{"layer_name": Tensor} that stores the current masks.
            scores: A dict{"layer_name": Tensor} that stores the pruning scores of weights.
            progressive_step: An integer representing the number of current step in progressive pruning.
            progressive_configs: A dict that stores configurations of progressive pruning.

        Returns:
            A dict{"layer_name": Tensor} that stores the masks generated in progressive pruning.
        """
        progressive_steps = progressive_configs["progressive_steps"]
        progressive_masks = {}
        new_added_masks = self.update_new_added_masks(pre_masks, cur_masks)
        for key in scores.keys():
            block_size = self.block_size[key]
            mask_num_each_block = progressive_step * int((block_size[0] * block_size[1]) / progressive_steps)
            new_added_filter = 1 - new_added_masks[key]
            score = scores[key]
            score_masked = (score * new_added_filter).abs()
            score_masked = self._reshape_orig_to_2dims(score_masked)

            # similar to n:m type
            # generate progressive masks from scores
            shape = score_masked.shape
            new_shape = [shape[0] // block_size[0], block_size[0], shape[1] // block_size[1], block_size[1]]
            score_masked_new = score_masked.clone()
            score_masked_new_shape = score_masked_new.reshape(new_shape)
            score_masked_new_shape = score_masked_new_shape.permute(0, 2, 1, 3)
            score_masked_new_flatten = torch.flatten(score_masked_new_shape, start_dim=2, end_dim=3)
            threshold, _ = torch.kthvalue(score_masked_new_flatten, mask_num_each_block, dim=2)
            threshold = threshold.unsqueeze(-1)
            threshold = threshold.repeat(1, 1, (block_size[0] * block_size[1]))
            threshold = threshold.reshape([threshold.shape[0], threshold.shape[1], block_size[0], block_size[1]])
            threshold = threshold.permute(0, 2, 1, 3)
            threshold = threshold.reshape((shape[0], shape[1]))
            one = torch.tensor([1.]).to(score.device)
            zero = torch.tensor([0.]).to(score.device)
            mask = torch.where(score_masked <= threshold, zero, one)

            mask = self._reshape_2dims_to_orig(mask, pre_masks[key].shape)
            progressive_mask = pre_masks[key] * (mask + new_added_masks[key])
            progressive_masks[key] = torch.where(progressive_mask == 0, zero, one)  # binary
        return progressive_masks

    def update_progressive_masks_local(self, pre_masks, cur_masks, scores, progressive_step, progressive_configs):
        """Generate progressive masks in a local pruning domain.
        
        Args:
            pre_masks: Dict{"layer_name": Tensor} that stores the masks generated after the last pruning step.
            cur_masks: Dict{"layer_name": Tensor} that stores the current masks.
            scores: A dict{"layer_name": Tensor} that stores the pruning scores of weights.
            progressive_step: An integer representing the number of current step in progressive pruning.
            progressive_configs: A dict that stores configurations of progressive pruning.
        
        Returns:
            A dict{"layer_name": Tensor} that stores the masks generated in progressive pruning.
        """
        progressive_type = progressive_configs["progressive_type"]
        if progressive_type == "linear":
            progressive_masks = self.update_progressive_masks_linear(pre_masks, cur_masks, \
                                                                     progressive_step, progressive_configs)
        elif progressive_type == "scores":
            progressive_masks = self.update_progressive_masks_scores(pre_masks, cur_masks, scores, \
                                                                     progressive_step, progressive_configs)
        else:
            raise NotImplementedError
        return progressive_masks

    def update_progressive_masks_global(self, pre_masks, cur_masks, scores, progressive_step, progressive_configs):
        """Gather all layer's scores to obtain a threshold that would be applied to all layers.
        
        Args:
            pre_masks: Dict{"layer_name": Tensor} that stores the masks generated after the last pruning step.
            cur_masks: Dict{"layer_name": Tensor} that stores the current masks.
            scores: A dict{"layer_name": Tensor} that stores the pruning scores of weights.
            progressive_step: An integer representing the number of current step in progressive pruning.
            progressive_configs: A dict that stores configurations of progressive pruning.
        
        Returns:
            A dict{"layer_name": Tensor} that stores the masks generated in progressive pruning.
        """
        progressive_steps = progressive_configs["progressive_steps"]
        progressive_masks = {}
        global_new_added_score_list = []
        new_added_masks = self.update_new_added_masks(pre_masks, cur_masks)
        new_added_masks_cnts = self.count_new_masked_cnts(new_added_masks)
        kth_masked_position = (new_added_masks_cnts * progressive_step) // progressive_steps
        for key in scores.keys():
            block_size = self.block_size[key]
            mask_num_each_block = progressive_step * int((block_size[0] * block_size[1]) // progressive_steps)
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
        for key in scores.keys():
            new_added_mask = new_added_masks[key]
            score = scores[key]
            new_added_filter = 1 - new_added_mask
            score_masked = (score * new_added_filter).abs()
            zero = torch.tensor([0.]).to(score.device)
            one = torch.tensor([1.]).to(score.device)
            progressive_mask = (new_added_mask + torch.where(score_masked <= threshold, zero, one)) * pre_masks[key]
            progressive_masks[key] = progressive_mask
        return progressive_masks


@register_pattern('N:M')
class PatternNInM(BasePattern):
    """Pruning Pattern.
    
    A Pattern class derived from Pattern. In this pattern, N out of every M continuous weights will be pruned.
    For more info of this pattern, please refer to :
    https://github.com/intel/neural-compressor/blob/master/docs/sparsity.md
    
    Args:
        config: A config dict object that contains the pattern information.
        
    Attributes:
        N: The number of elements to be pruned in a weight sequence.
        M: The size of the weight sequence.
    """

    def __init__(self, config, modules):
        """Initialize the basic pruning unit of N:M pattern."""
        super(PatternNInM, self).__init__(config, modules)
        pattern = self.pattern.split('_')[-1]
        self.N = int(pattern.split(':')[0])
        self.M = int(pattern.split(':')[1])  ##m is bigger
        self.check_layer_validity(self.modules, (self.N, self.M))

    def check_layer_validity(self, datas: dict, block_size: tuple):
        """Check if a layer is valid for this block_size.
        
        Args:
            datas: A dict object containing the weights for all layers.
            block_size: A tuple representing the size of the pattern block.
        """
        self.invalid_layers = []
        for key in datas.keys():
            data = datas[key].weight
            data = self._reshape_orig_to_2dims(data)
            shape = data.shape
            if shape[1] % block_size[1] != 0:
                self.invalid_layers.append(key)
                logger.warning(f"{key} shape {shape} cannot be divided by {self.pattern}")

    def get_reduced_masks_from_data(self, data, key):
        """Obtain the unpruned weights and reshape according to the block_size.
        
        Args:
            data: Input.
            key: The layer name.

        Returns:
            A tensor representing the unpruned weights.
        """
        data = self._reshape_orig_to_2dims(data)
        shape = data.shape
        M = self.M
        N = self.N
        new_shape = [shape[0], shape[1] // M, M]
        data = data.reshape(new_shape)
        nonzeros = torch.count_nonzero(data, dim=-1)
        reduced_mask = nonzeros > N
        return reduced_mask

    def get_least_ninm_mask_from_data(self, score):
        """Generate the least N scores in M.
        
        Args:
            score: the pruning scores of weights.

        Returns:
            A dict with the identical size as pre_masks and its 0/1 values are updated. 
                1 means unpruned and 0 means pruned.
        """
        current_score = score
        M = self.M
        N = self.N
        current_score = self._reshape_orig_to_2dims(current_score)
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
        return mask

    def get_sparsity_ratio(self, pre_masks, return_dict=False):
        """Please note that the zero cnt and total cnt are all block_wise for supporting channel-wise pruning.

        The return sparsity ratio is elementwised.
        
        Args:
            pre_masks: Dict{"layer_name": Tensor} that stores the masks generated after the last pruning step.
            return_dict: A bool determining whether to return more information like zero_cnt and total_cnt.
            
        Returns:
            An elementwise sparisty ratio.
        """
        zero_cnt = 0
        total_cnt = 0
        for key in pre_masks.keys():
            if key in self.invalid_layers:
                # total_cnt += pre_masks[key].numel() // self.M
                continue
            reduced_mask = self.get_reduced_masks_from_data(pre_masks[key], key)
            zero_cnt += int((torch.sum(reduced_mask == 0)).data.item())
            total_cnt += int(reduced_mask.numel())
        sparsity_ratio = float(zero_cnt) / total_cnt * self.N / self.M

        if return_dict:
            return {"sparsity_ratio": sparsity_ratio, "zero_cnt": zero_cnt,
                    "total_cnt": total_cnt}
        else:
            return sparsity_ratio

    def _reshape_orig_to_2dims(self, data):
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

    def _reshape_2dims_to_orig(self, data, orig_shape):
        """Recover layers that are not two-dimensional(e.g conv layer).

        Args:
            data: Input.
            
        Returns:
            Reshaped data.
        """
        if len(orig_shape) == 4:
            data = data.reshape(orig_shape[0], orig_shape[2], orig_shape[3], orig_shape[1])
            data = data.permute(0, 3, 1, 2)
        return data

    def reshape_orig_to_pattern(self, data, key):
        """Reshape the data based on the pruning pattern.
        
        Args:
            data: Input.
            key: layer name.
        
        Returns:
            Reshaped data.
        """
        data = self._reshape_orig_to_2dims(data)
        shape = data.shape
        new_shape = [shape[0], shape[1] // self.M, self.M]
        data = data.reshape(new_shape)
        return data

    def reshape_reduced_to_orig(self, data, key, orig_shape):
        """Reshape the reduced data to its original shape.
        
        Args:
            data: Input.
            key: The layer name.
            orig_shape: The original shape of the layer.
            
        Returns:
            Data of its original shape.
        """
        data = data.repeat_interleave(self.M, dim=-1)
        return self._reshape_2dims_to_orig(data, orig_shape)

    def reduce_scores(self, scores):
        """Calculate the pruning scores after reducing the data and obtain the least N scores in M.
        
        Args:
            scores: Pruning scores of weights.

        Returns:
            Updated pruning scores and the least N scores in M.
        """
        ##to get the least N scores in M
        M = self.M
        N = self.N
        least_ninm_masks = {}
        new_scores = {}
        for key in scores.keys():
            if key in self.invalid_layers:
                continue
            if self.keep_mask_layers.get(key, False):
                continue
            current_score = scores[key]
            mask = self.get_least_ninm_mask_from_data(current_score)
            current_score_new = self._reshape_orig_to_2dims(current_score)
            shape = current_score_new.shape
            current_score_new = current_score_new.reshape((shape[0], shape[1]))
            ##to get the sum of N scores in each block with M
            current_score_new = current_score_new * (1.0 - mask)
            current_score_new = current_score_new.reshape(shape[0], shape[1] // M, M)
            score_sum = self.reduce_tensor(current_score_new, dim=-1)
            least_ninm_masks[key] = mask
            new_scores[key] = score_sum
        return new_scores, least_ninm_masks

    def get_ele_mask_per_threshold(self, score, threshold, block_size, least_ninm_mask):
        """Get the elementwise mask per threshold.

        Args:
            score: A tensor that stores the pruning scores of weights.
            threshold: A float used to determine whether to prune a weight.
            block_size: A list of two integers representing the height and width of the block.
            least_m_in_m_masks: A tensor representing the least N scores in M.
            
        Returns:
            mask: The elementwise pruning mask.
        """
        zero = torch.tensor([0.]).to(score.device)
        one = torch.tensor([1.]).to(score.device)
        mask = torch.where(score <= threshold, zero, one)
        mask = mask.repeat_interleave(block_size[1], dim=-1)
        ## both zero will be zero
        mask = (mask + least_ninm_mask)
        mask = torch.where(mask <= 0, zero, one)
        return mask

    def get_masks_global(self, scores, cur_target_sparsity_ratio, pre_masks,
                         keep_exact_sparsity_ratio=True):
        """Generate masks for layers.
        
        Gather all layer's scores together and calculate a common threshold.
        This threshold will be applied for all layers.
        
        Args:
            scores: A dict{"layer_name": Tensor} that stores the pruning scores of weights.
            target_sparsity_ratio: A float representing the model's final sparsity.
            pre_masks: A dict{"layer_name": Tensor} representing the masks generated after the last pruning step.
            max_sparsity_ratio_per_op: A float representing the maximum sparsity that one layer can reach.
            
        Returns:
            A dict with the identical size as pre_masks and its 0/1 values are updated. 
                1 means unpruned and 0 means pruned.
        """
        masks = pre_masks

        block_sparsity_ratio = cur_target_sparsity_ratio * self.M / self.N
        k_blockwise = self.update_residual_cnt(pre_masks, block_sparsity_ratio)
        if k_blockwise <= 0:
            return masks
        new_scores, least_ninm_masks = self.reduce_scores(scores)
        global_scores = torch.cat([torch.flatten(v) for v in new_scores.values()])  ##block_wise
        residual_k = k_blockwise
        not_exceed_layers = [key for key in new_scores.keys()]

        while True:
            threshold, _ = torch.kthvalue(global_scores, residual_k)
            for key in not_exceed_layers:
                score = new_scores[key]
                mask = self.get_ele_mask_per_threshold(score, threshold, (self.N, self.M), least_ninm_masks[key])
                info = self.get_sparsity_ratio({key: mask}, return_dict=True)
                zero_cnt = info["zero_cnt"]
                total_cnt = info["total_cnt"]
                current_sparsity_ratio = float(zero_cnt) / total_cnt
                key_new_sparsity = SparsityInfo(zero_cnt, total_cnt, current_sparsity_ratio)
                need_adjust, adjust_ratio = self.adjust_ratio(masks, key, key_new_sparsity,
                                                              self.max_sparsity_ratio_per_op * self.M / self.N,
                                                              self.min_sparsity_ratio_per_op * self.M / self.N,
                                                              self.target_sparsity_ratio * self.M / self.N)

                if need_adjust:
                    self.keep_mask_layers[key] = True
                    masks[key] = self.get_single_mask_per_target_ratio(new_scores[key], adjust_ratio)
                    masks[key] = masks[key].repeat_interleave(self.M, dim=-1)
                    ## both zero will be zero
                    masks[key] = (masks[key] + least_ninm_masks[key])
                    zero = torch.tensor([0.]).to(score.device)
                    one = torch.tensor([1.]).to(score.device)
                    masks[key] = torch.where(masks[key] <= 0, zero, one)
                    if keep_exact_sparsity_ratio:
                        zero_cnt = self.get_sparsity_ratio({key: masks[key]}, return_dict=True)["zero_cnt"]
                        residual_k -= zero_cnt
                else:
                    masks[key] = mask
            if not keep_exact_sparsity_ratio:
                break
            new_not_exceed_layers = [key for key in new_scores.keys() if not self.keep_mask_layers.get(key, False)]
            if not_exceed_layers == new_not_exceed_layers or len(new_not_exceed_layers) == 0:
                break
            not_exceed_layers = new_not_exceed_layers
            global_scores = torch.cat([torch.flatten(new_scores[key]) for key in not_exceed_layers])

        for key in masks.keys():
            if key in self.invalid_layers:
                continue
            if len(scores[key].shape) == 4:  ## need to permute
                mask = masks[key]
                orig_shape = scores[key].shape
                mask = self._reshape_2dims_to_orig(mask, orig_shape)
                masks[key] = mask
            layer_ratio = torch.sum(masks[key] == 0.0).data.item() / masks[key].numel()
            logger.info(f'layer {key} sparsity_ratio is {layer_ratio}')
        return masks

    def get_pattern_lock_masks(self, modules):
        """Obtain masks from original weight map, by masking where weights' are zero.
        
        Args:
            modules: A dict{"layer_name": Tensor} that stores weights.
            
        Returns:
            A dict with the identical size as modules, containing pattern lock masks.
        """
        pattern_lock_masks = {}
        for key in modules.keys():
            weight = modules[key].weight
            orig_shape = weight.shape
            if key in self.invalid_layers:
                mask = torch.ones(orig_shape, device=weight.device)
                pattern_lock_masks[key] = mask
                continue
            mask = self.get_least_ninm_mask_from_data(weight)
            mask = self._reshape_2dims_to_orig(mask, orig_shape)
            pattern_lock_masks[key] = mask
        return pattern_lock_masks
