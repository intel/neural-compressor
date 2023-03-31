"""Weight Squeezer."""
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

import torch
import torch.nn as nn
import random
from ..utils import logger

class PostCompressionUtils(object):
    """Operations library related to weight compression."""

    @staticmethod
    def obtain_output_masks(tensor):
        """Obtain a lower dimension mask for an sparse weight matrix."""
        # dim 1 is input channel
        tensor_reduce = torch.sum(tensor.abs(), 1)
        mask_reduce = torch.where(tensor_reduce==0.0, 0, 1)
        return mask_reduce

    @staticmethod
    def obtain_input_masks(tensor):
        """Obtain a lower dimension mask for an sparse weight matrix."""
        # dim 0 is output channel
        tensor_reduce = torch.sum(tensor.abs(), 0)
        mask_reduce = torch.where(tensor_reduce==0.0, 0, 1)
        return mask_reduce

    @staticmethod
    def get_mha_output_indice(tensor, hidden_size, head_nums):
        """Obtain a lower dimension mask for an sparse weight matrix."""
        head_size = hidden_size // head_nums
        tensor_reduce = torch.sum(tensor.abs(), 1)
        mask_reduce = torch.where(tensor_reduce==0.0, 0, 1)
        mask_reduce_headwise = mask_reduce.reshape(mask_reduce.shape[0] // head_size, head_size).sum(1) / head_size
        mask_reduce_indice = torch.nonzero(torch.where(mask_reduce_headwise <= 0.00001, 1, 0) == 1).squeeze().tolist()
        if isinstance(mask_reduce_indice, int): 
            # only one channel is pruned
            return [mask_reduce_indice]
        return mask_reduce_indice

    @staticmethod
    def get_mha_input_indice(tensor, hidden_size, head_nums):
        """Obtain a lower dimension mask for an sparse weight matrix."""
        head_size = hidden_size // head_nums
        tensor_reduce = torch.sum(tensor.abs(), 0)
        mask_reduce = torch.where(tensor_reduce==0.0, 0, 1)
        mask_reduce_headwise = mask_reduce.reshape(mask_reduce.shape[0] // head_size, head_size).sum(1) / head_size
        mask_reduce_indice = torch.nonzero(torch.where(mask_reduce_headwise <= 0.00001, 1, 0) == 1).squeeze().tolist()
        if isinstance(mask_reduce_indice, int): 
            # only one channel is pruned
            return [mask_reduce_indice]
        return mask_reduce_indice

    @staticmethod
    def get_mask_indices(mask):
        """Obtain the masks' 0 elements' indice."""
        # return the indices of elements whose values are zero
        indice = torch.nonzero(torch.where(mask == 0, 1, 0) == 1).squeeze().tolist()
        if isinstance(indice, int):
            # only one channel is to pruned
            return [indice]
        return indice

    @staticmethod
    def find_pruneable_indices(indice, n_heads, head_size=1, round_option=0):
        """Obtain the masks' 0 elements' indice.

        Args:
            indices: list(int): channel / heads' indices to be pruned
            n_head: int. number of head to prune
            head_size: for head pruning, it is head size, for channel pruning, it is 1
            round_option: if pruning channel number does not equals to 32x, (16x), round it to a 32x int
        
        Return:
            indice: the mask' zero-value elements indice
            indice_to_keep: the masks one-value elements indice
        """
        # import pdb;pdb.set_trace()
        # round the pruning number to be a multiple of 2^n
        if round_option > 0:
            round_length = (indice.__len__() // round_option) * round_option
            indice = random.sample(indice, round_length)
            indice.sort()
        else:
            # round_option is 0, means no need to select indice
            pass

        mask = torch.ones(n_heads, head_size)
        already_pruned_heads = set()
        indice = set(indice)  # Convert to set and remove already pruned heads
        for head in indice:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        indice_to_keep: torch.LongTensor = torch.arange(len(mask))[mask].long()
        return indice, indice_to_keep

    @staticmethod
    def prune_linear(layer, index, device, dim = 0, prune_bias = True):
        """Operation to compress a sparse linear layer's weight.
        
        Args:
            layer (torch.nn.Linear): a linear layer
            index (list): the indice which channels/heads should be kept
            dim (int): input channel (1) or output channel (0)
            prune_bias (bool): if output channel is pruned, bias should also be pruned.

        Return:
            The same layer object whose weight (probability also bias) has been compressed. 
        """
        # import pdb;pdb.set_trace()
        index = index.to(device)
        _w = layer.weight.index_select(dim, index).clone().detach()
        if prune_bias:
            _b = layer.bias[index].clone().detach()
        else:
            _b = layer.bias.clone().detach()
        new_size = list(layer.weight.size())
        new_size[dim] = len(index)
        setattr(layer, "in_features", new_size[1])
        setattr(layer, "out_features", new_size[0])
        setattr(layer, "weight", torch.nn.Parameter(_w.clone()))
        setattr(layer, "bias", torch.nn.Parameter(_b.clone()))

        # new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None)
        layer.weight.requires_grad = False
        layer.weight.copy_(_w.contiguous())
        layer.weight.requires_grad = True

        if prune_bias:
            layer.bias.requires_grad = False
            layer.bias.copy_(_b.contiguous())
            layer.bias.requires_grad = True

class LinearCompression(object):
    """Class which automatically compresses two consecutive linear layers.

    For two consecutive linear layer, when the second layer's input channel is pruned, 
    then the first layer's output channel can also be pruned, 
    while the second layer's output hidden state value is identical. 
    for example, two consecutive linears have following structure.
    x = layer_1(input)
    x = act_fn(x)
    x = layer_2(x)

    Args:
        layer_1 (torch.nn.Linear): the first Linear layer.
        layer_2 (torch.nn.Linear): the second Linear layer.

    Attributes:
        layer_1 (torch.nn.Linear): the first Linear layer.
        layer_2 (torch.nn.Linear): the second Linear layer.
        device: the device of layers' weights.
    """

    def __init__(self, layer_1, layer_2):
        """Initialize."""
        assert type(layer_1).__name__ == "Linear", "layer 1 should be torch.nn.modules.linear.Linear module type"
        assert type(layer_2).__name__ == "Linear", "layer 1 should be torch.nn.modules.linear.Linear module type"
        self.layer_1 = layer_1
        self.layer_2 = layer_2
        self.device = self.layer_1.weight.device
        self.log = {
            '1_before': [self.layer_1.out_features, self.layer_1.in_features],
            '2_before': [self.layer_2.out_features, self.layer_2.in_features],
        }
    
    def __call__(self, mask=None, round_value=0):
        """Operation to execute weight compression process.

        Args:
            mask: the predefined mask of the second layer's input channel.
                  if is None, the API automatically detects the sparse channel and generates the mask.
            round_value (int): if pruning channel number does not equals to 32x, (16x), round it to a 32x int

        """
        if mask is not None:
            layer_2_mask = mask.clone().to(self.device)
        else:
            layer_2_mask = PostCompressionUtils.obtain_input_masks(self.layer_2.weight)
        layer_2_indice_to_prune = PostCompressionUtils.get_mask_indices(layer_2_mask)
        _, layer_2_indice_to_keep = PostCompressionUtils.find_pruneable_indices(
            layer_2_indice_to_prune, 
            self.layer_2.in_features, 1, round_value
        ) # 1 refer to channel-wise pruning
        PostCompressionUtils.prune_linear(
            self.layer_2, 
            layer_2_indice_to_keep, 
            device=self.device, dim=1, prune_bias=False
        )
        PostCompressionUtils.prune_linear(
            self.layer_1, 
            layer_2_indice_to_keep, 
            device=self.device, dim=0, prune_bias=True
        )
        # Summary:
        self.log['1_after'] = [self.layer_1.out_features, self.layer_1.in_features]
        self.log['2_after'] = [self.layer_2.out_features, self.layer_2.in_features]
        logger.info(f"layer_1 compression: {self.log['1_before']} -> {self.log['1_after']}")
        logger.info(f"layer_2 compression: {self.log['2_before']} -> {self.log['2_after']}")

class LinearCompressionIterator(object):
    """Pruner of a sequence of consecutive linear patterns.

    Attributes:
        linear_patterns (dict/list): a iterable object of consecutive linear patterns.
    """

    def __init__(self, linear_patterns):
        """Initialize."""
        self.linear_patterns = linear_patterns

    def __call__(self, masks=None, round_value=0):
        """Operation to execute weight compression process.

        Args:
            mask: the predefined masks of the second layers input channel.
                  if is None, the API automatically detects the sparse channel and generates the mask.
            round_value (int): if pruning channel number does not equals to 32x, (16x), round it to a 32x int

        """
        # masks should have same length as layers patterns
        # self.linear_patterns: a list or dict
        if isinstance(masks, list):
            mask_len = len(masks)
        if isinstance(masks, torch.Tensor):
            mask_len = masks.shape[0]
        layer_idx = 0
        for pattern in self.linear_patterns:
            if isinstance(self.linear_patterns, dict):
                linear_pruner = LinearCompression(self.linear_patterns[pattern][0], self.linear_patterns[pattern][1])
            elif isinstance(self.linear_patterns, list):
                linear_pruner = LinearCompression(pattern[0], pattern[1])
            else:
                raise NotImplementedError
            # compression
            if masks != None and layer_idx < len(masks):
                linear_pruner(mask=masks[layer_idx], round_value=round_value)
            else:
                linear_pruner(round_value=round_value)
            layer_idx += 1
            del linear_pruner
        logger.info(f"Post pruning model slim finished.")

class MHACompression(object):
    """
    use huggingface build-in head mask to compress weights,
    (one mha module only have one head mask), tested on training-free progress
    different layers have different masks, but heads number to mask is the same
    """
    def __init__(self, mha, head_nums, head_size):
        # for example, bert-base hidden-size is 768, head_nums is 12, then head_size is 64
        # four linear layers to be pruned
        self.mha = mha
        self.query = self.mha.self.query
        self.key = self.mha.self.key
        self.value = self.mha.self.value
        self.ffn = self.mha.output.dense
        # other configs
        # hidden_size = head_nums * head_size
        self.head_nums = head_nums
        self.head_size = head_size
        self.hidden_size = int(self.head_nums * self.head_size)
        # device
        self.device = self.query.weight.device
    
    def find_common_indice(self, d):
        common_indice = list(d.values())[0]
        for k, v in d.items():
            common_indice = set(common_indice) & set(v)
        return list(common_indice)
    
    def __call__(self, head_mask = None):
        """
        for qkv, prune output channel, for output, prune input channel
        four linear shares identical masks (attention mask)
        """
        query_indice = PostCompressionUtils.get_mha_output_indice(self.query.weight, self.hidden_size, self.head_nums)
        key_indice = PostCompressionUtils.get_mha_output_indice(self.key.weight, self.hidden_size, self.head_nums)
        value_indice = PostCompressionUtils.get_mha_output_indice(self.value.weight, self.hidden_size, self.head_nums)
        ffn_indice = PostCompressionUtils.get_mha_input_indice(self.ffn.weight, self.hidden_size, self.head_nums)
        all_indice_to_prune = {
            "query": query_indice,
            "key": key_indice,
            "value": value_indice,
            "ffn": ffn_indice
        }
        
        # alignment, take the least heads to prune
        logger.info(all_indice_to_prune)
        prune_indice = self.find_common_indice(all_indice_to_prune)
        logger.info(prune_indice)
        # 1 refer to channel-wise pruning
        _, indice_to_keep = PostCompressionUtils.find_pruneable_indices(prune_indice, self.head_nums, self.head_size)
        # prune qkv, outputs
        # Prune linear layers
        PostCompressionUtils.prune_linear(self.query, indice_to_keep, self.device, dim=0, prune_bias=True)
        PostCompressionUtils.prune_linear(self.key, indice_to_keep, self.device, dim=0, prune_bias=True)
        PostCompressionUtils.prune_linear(self.value, indice_to_keep, self.device, dim=0, prune_bias=True)
        PostCompressionUtils.prune_linear(self.ffn, indice_to_keep, self.device, dim=1, prune_bias=False)

        # Update hyper params and store pruned heads
        self.mha.self.num_attention_heads = self.mha.self.num_attention_heads - len(prune_indice)
        self.mha.self.all_head_size = self.mha.self.attention_head_size * self.mha.self.num_attention_heads
