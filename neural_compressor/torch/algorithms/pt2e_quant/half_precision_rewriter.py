# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
from torch.fx import subgraph_rewriter
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.utils import matcher_utils
from torch.fx.subgraph_rewriter import Match
from typing_extensions import TypeAlias

from neural_compressor.common import utils

# =============================================================================
# Search and replace patterns
# =============================================================================
TorchFuncType: TypeAlias = Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor]


@dataclass
class PatternPair:
    search_pattern: torch.fx.GraphModule
    replace_pattern: torch.fx.GraphModule
    match_filters: Optional[List[Callable[[matcher_utils.InternalMatch, torch.fx.Graph, torch.fx.Graph], bool]]]


# key: torch func
# value: the tuple of args
FuncArgsMappingType: TypeAlias = Dict[TorchFuncType, Tuple[torch.Tensor, ...]]


# Align with https://pytorch.org/docs/stable/amp.html#cpu-ops-that-can-autocast-to-bfloat16
# TODO: complete the mapping
FN_ARGS_MAPPING: FuncArgsMappingType = {
    torch.nn.functional.linear: (torch.randn(0, 0), torch.randn(0, 0)),  # linear w/o bias
    torch.nn.functional.linear: (torch.randn(0, 0), torch.randn(0, 0), torch.randn(0)),  # linear w/ bias
}
# TODO: complete the mapping
FN_ATEN_OPS_MAPPING = {
    torch.nn.functional.linear: torch.ops.aten.linear.default,
}

SUPPORTED_OPERATORS = FN_ATEN_OPS_MAPPING.values()


PatternRegistryType: TypeAlias = Dict[TorchFuncType, PatternPair]
HALF_PRECISION_PATTERN_REGISTRY: Dict[torch.dtype, PatternRegistryType] = {torch.float16: {}, torch.bfloat16: {}}

# FP16_PATTERN_REGISTRY: PatternRegistryType = HALF_PRECISION_PATTERN_REGISTRY[torch.float16]
# BF16_PATTERN_REGISTRY: PatternRegistryType = HALF_PRECISION_PATTERN_REGISTRY[torch.bfloat16]
# ALL_NODES = "ALL_NODES"


def is_target_node_in_candidate_list(match, original_graph, pattern_graph, node_list, target_op):
    """Filter the node with target operator in match and check if it is in `node_list`."""
    target_node = None
    for node in pattern_graph.nodes:
        if node.target == target_op:
            target_node = node
            break
    if target_node is None:
        return False
    matched_node = match.nodes_map[target_node]
    return matched_node in node_list


def pattern_factory(fn: TorchFuncType, fn_arg: Tuple[torch.Tensor, ...], target_dtype: torch.dtype = torch.float16):
    """Create a search, replace pattern and filter functions for a given torch function and its arguments."""
    assert target_dtype in [
        torch.float16,
        torch.bfloat16,
    ], f"target_dtype should either be `torch.float16` or `torch.bfloat16`, but got {target_dtype}"

    def replace_fn_wrapper(fn_args, fn):
        converted_args = [arg.to(target_dtype) for arg in fn_args]
        target_dtype_out = fn(*converted_args)
        return target_dtype_out.float()

    replace_fn = partial(replace_fn_wrapper, fn=fn)

    search_pattern_gm = make_fx(fn, pre_dispatch=True)(*fn_arg)
    # TODO: double-check `*fn_args` or `fn_args`
    replace_pattern_gm = make_fx(replace_fn, pre_dispatch=True)(fn_arg)
    target_op = FN_ATEN_OPS_MAPPING[fn]
    filter_fn = partial(is_target_node_in_candidate_list, target_op=target_op)
    pattern_pair = PatternPair(search_pattern_gm, replace_pattern_gm, [filter_fn])

    return pattern_pair


def _register_pattern_pair(dtype: torch.dtype) -> None:
    for fn, fn_args in FN_ARGS_MAPPING.items():
        pattern_pair = pattern_factory(fn, fn_args)
        HALF_PRECISION_PATTERN_REGISTRY[dtype][fn] = pattern_pair
    utils.logger.info(
        f"Registered {len(HALF_PRECISION_PATTERN_REGISTRY[dtype])} search and replace patterns for {dtype}."
    )


_register_pattern_pair(torch.float16)


def apply_single_pattern_pair(gm: torch.fx.GraphModule, pattern_pair: PatternPair, node_list):
    match_filters_with_node_list = []
    if pattern_pair.match_filters:
        match_filters_with_node_list = [
            partial(filter_fn, node_list=node_list) for filter_fn in pattern_pair.match_filters
        ]
    match_and_replacements = subgraph_rewriter.replace_pattern_with_filters(
        gm=gm,
        pattern=pattern_pair.search_pattern,
        replacement=pattern_pair.replace_pattern,
        match_filters=match_filters_with_node_list,
    )
    utils.logger.info(f"Found {len(match_and_replacements)} matches.")

    match_list = [Match(anchor=m.anchor, nodes_map=m.nodes_map) for m in match_and_replacements]
    return match_list


def get_unquantized_node_list(gm: torch.fx.GraphModule):
    unquantized_node_list = []
    for node in gm.graph.nodes:
        if meta := getattr(node, "meta"):
            if quantization_annotation := meta.get(xiq.QUANT_ANNOTATION_KEY):
                if quantization_annotation._annotated:
                    continue
        unquantized_node_list.append(node)
    return unquantized_node_list


def get_half_precision_node_list(gm, user_specific_node_list: List[str]):
    """Intersection between `unquantized_node_list` and `user_specific_node_list`"""
    # TODO: implement it, current return all unquantized_node_list
    half_precision_node_list = []
    unquantized_node_list = get_unquantized_node_list(gm)
    for node in unquantized_node_list:
        if node.target in SUPPORTED_OPERATORS:
            half_precision_node_list.append(node)
    utils.logger.info(f"Found {len(half_precision_node_list)} nodes to convert to half precision.")
    return half_precision_node_list


def transformation(gm: torch.fx.GraphModule, node_candidate_list: List[str], target_dtype: torch.dtype = torch.float16):
    """Convert the nodes in `node_candidate_list` to `target_dtype` if possible."""
    for pattern_pair in HALF_PRECISION_PATTERN_REGISTRY[target_dtype].values():
        apply_single_pattern_pair(gm, pattern_pair, node_candidate_list)
    print(gm.print_readable())
