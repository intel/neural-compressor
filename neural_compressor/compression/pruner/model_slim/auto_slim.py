"""Auto slim."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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
# model slim related

from ..utils import logger


def model_slim(model, dataloader=None, round_multiplier=32):
    """Slim the sparse model automatically."""
    try:
        model = model_slim_ffn2(model, dataloader, round_multiplier)
    except:
        logger.warning("model Linear2Linear slim failed.")
    try:
        model = model_slim_mha(model, dataloader)
    except:
        logger.warning("model MHA slim failed.")
    return model


def model_slim_ffn2(model, dataloader=None, round_multiplier=32):
    """Remove some sparse part in the model permanently and obtain acceleration directly.

    Args:
        model: a sprase model.
        round_multiplier(int): the channel number after slimming should be multiple of this number.
    """
    from .pattern_analyzer import Linear2LinearSearcher
    from .weight_slim import LinearCompressionIterator

    logger.warning("You are using model slim methods, some weight channels will be removed permanently.")
    pa_obj = Linear2LinearSearcher(model, dataloader)
    layers = pa_obj.search()
    layers = pa_obj.from_layer_name_to_object(layers)
    linear_pruner = LinearCompressionIterator(layers)
    linear_pruner(masks=None, round_value=round_multiplier)
    return model


def model_slim_mha(model, dataloader=None):
    """Remove some sparse part in the model permanently and obtain acceleration directly.

    Args:
        model: a sprase model.
    """
    from .pattern_analyzer import SelfMHASearcher
    from .weight_slim import MHACompression

    logger.warning("You are using model slim methods, some attention heads will be removed permanently.")
    pa_obj = SelfMHASearcher(model, dataloader)
    layers, _ = pa_obj.search(split_qkv_ffn=False)
    layers = pa_obj.obtain_mha_module(layers)
    layers = pa_obj.from_layer_name_to_object(layers)
    for layer in layers:
        mha_compression = MHACompression(layer)
        mha_compression()
    return model


# auto slim config
def parse_auto_slim_config(model, dataloader=None, ffn2_sparsity=0.0, mha_sparsity=0.0, **kwargs):
    """Get model slim pruning configs."""
    auto_slim_configs = []
    if ffn2_sparsity > 0 and ffn2_sparsity < 1:
        auto_slim_configs += generate_ffn2_pruning_config(model, dataloader, ffn2_sparsity, **kwargs)
    if mha_sparsity > 0 and mha_sparsity < 1:
        auto_slim_configs += generate_mha_pruning_config(model, dataloader, mha_sparsity, **kwargs)
    return auto_slim_configs


def generate_ffn2_pruning_config(model, dataloader, ffn2_sparsity, **kwargs):
    """Get consecutive linear layers pruning configs."""
    from .pattern_analyzer import Linear2LinearSearcher

    searcher = Linear2LinearSearcher(model, dataloader)
    layers = searcher.search()
    # extract the second linear layer
    ffn_layers = [ffn2_module["root_linear"] for ffn2_module in layers]
    ffn2_pruning_config = [{"op_names": ffn_layers, "pattern": "channelx1", "target_sparsity": ffn2_sparsity}]
    # append kwargs to generated config
    for item in ffn2_pruning_config:
        item.update(kwargs)
    return ffn2_pruning_config


def generate_mha_pruning_config(model, dataloader, mha_sparsity, **kwargs):
    """Get multi-head attention layers pruning configs."""
    # method 1: apply real mha pruning
    mha_pruning_config = [{"pattern": "mha", "target_sparsity": mha_sparsity}]
    # append kwargs to generated config
    for item in mha_pruning_config:
        item.update(kwargs)
    return mha_pruning_config

    # method 2: apply experimental mha pruning
    # from .pattern_analyzer import SelfMHASearcher
    # searcher = SelfMHASearcher(model, dataloader)
    # qkv_pattern, ffn_pattern = searcher.get_head_pattern()
    # qkv_layers, ffn_layers = searcher.search()
    # mha_pruning_config = [
    #     {
    #         "op_names": qkv_layers,
    #         "pattern": qkv_pattern,
    #         "target_sparsity": mha_sparsity,
    #     },
    #     {
    #         "op_names": ffn_layers,
    #         "pattern": ffn_pattern,
    #         "target_sparsity": mha_sparsity,
    #     }
    # ]
    # # append kwargs to generated config
    # for item in mha_pruning_config:
    #     item.update(kwargs)
    # return mha_pruning_config
