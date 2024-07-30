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
"""Rewrite the FP32 operators to FP16 or BF16 operators."""

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
import torch.ao.quantization.quantizer.xnnpack_quantizer as xpq
from torch.fx import subgraph_rewriter
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.subgraph_rewriter import Match
from typing_extensions import TypeAlias

from neural_compressor.common import utils

# =============================================================================
# Search and replace patterns
# =============================================================================
TorchFuncType: TypeAlias = Callable[..., Any]


@dataclass
class PatternPair:
    """Represents a pair of patterns used for search and replacement in a graph.

    Attributes:
        fn (TorchFuncType): The function type associated with the pattern pair.
        search_pattern (torch.fx.GraphModule): The search pattern to be matched in the graph.
        replace_pattern (torch.fx.GraphModule): The replacement pattern to be used when a match is found.
    """

    fn: TorchFuncType
    search_pattern: torch.fx.GraphModule
    replace_pattern: torch.fx.GraphModule


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

    pattern_pair = PatternPair(fn, search_pattern_gm, replace_pattern_gm)

    return pattern_pair


def _register_pattern_pair(dtype: torch.dtype) -> None:
    for fn, fn_args in FN_ARGS_MAPPING.items():
        pattern_pair = pattern_factory(fn, fn_args)
        HALF_PRECISION_PATTERN_REGISTRY[dtype][fn] = pattern_pair
    utils.logger.info(
        f"Registered {len(HALF_PRECISION_PATTERN_REGISTRY[dtype])} search and replace patterns for {dtype}."
    )


_register_pattern_pair(torch.float16)


def get_filter_fn(node_list, fn):
    """Filter function to check if a node with the target operator is in the given `node_list`.

    Args:
        node_list (list): List of nodes to check against.
        fn (str): Target operator.

    Returns:
        bool: True if the node with the target operator is in the `node_list`, False otherwise.
    """
    target_op = FN_ATEN_OPS_MAPPING[fn]

    def is_target_node_in_candidate_list(match, original_graph, pattern_graph):
        """Filter the node with target operator in match and check if it is in `node_list`."""
        target_node = None
        for node in pattern_graph.nodes:  # pragma: no cover
            if node.target == target_op:
                target_node = node
                break
        if target_node is None:  # pragma: no cover
            return False
        matched_node = match.nodes_map[target_node]
        return matched_node in node_list

    return is_target_node_in_candidate_list


def apply_single_pattern_pair(gm: torch.fx.GraphModule, pattern_pair: PatternPair, node_list):
    """Applies a single pattern pair to a given GraphModule.

    Args:
        gm (torch.fx.GraphModule): The GraphModule to apply the pattern pair to.
        pattern_pair (PatternPair): The pattern pair containing the search and replace patterns.
        node_list: The list of nodes to filter for pattern matching.

    Returns:
        List[Match]: A list of Match objects representing the matches found after applying the pattern pair.
    """
    filter_fn = get_filter_fn(node_list, pattern_pair.fn)
    match_and_replacements = subgraph_rewriter.replace_pattern_with_filters(
        gm=gm,
        pattern=pattern_pair.search_pattern,
        replacement=pattern_pair.replace_pattern,
        match_filters=[filter_fn],
    )
    utils.logger.info(f"Found {len(match_and_replacements)} matches.")

    match_list = [Match(anchor=m.anchor, nodes_map=m.nodes_map) for m in match_and_replacements]
    return match_list


def get_unquantized_node_set(gm: torch.fx.GraphModule):
    """Retrieves the set of unquantized nodes from a given GraphModule.

    Args:
        gm (torch.fx.GraphModule): The GraphModule to retrieve unquantized nodes from.

    Returns:
        set: A set containing the unquantized nodes.
    """
    unquantized_node_set = set()
    for node in gm.graph.nodes:
        if meta := getattr(node, "meta"):
            if quantization_annotation := meta.get(xiq.QUANT_ANNOTATION_KEY):
                none_annotation = xiq._X86InductorQuantizationAnnotation(_annotated=True)
                if quantization_annotation != none_annotation:  # pragma: no cover
                    continue
        unquantized_node_set.add(node)
    return unquantized_node_set


def transformation(gm: torch.fx.GraphModule, node_candidate_list: List[str], target_dtype: torch.dtype = torch.float16):
    """Convert the nodes in `node_candidate_list` to `target_dtype` if possible."""
    for pattern_pair in HALF_PRECISION_PATTERN_REGISTRY[target_dtype].values():
        apply_single_pattern_pair(gm, pattern_pair, node_candidate_list)
    utils.logger.info("Half precision conversion is done:")
    if utils.level_name == "DEBUG":  # pragma: no cover
        gm.print_readable(True)


# =============================================================================
# Utils to parse the node candidate set for half precision conversion
# =============================================================================


def _parse_node_candidate_set_from_user_config(config, gm):
    """Parse the node candidate set from user config."""
    op_type_configs, op_name_configs = config._get_op_name_op_type_config()
    op_type_filters = []
    op_name_filters = []
    for op_type_name, config in op_type_configs.items():  # pragma: no cover
        op_type = getattr(torch.nn, op_type_name)
        if config.act_dtype == "fp16":  # pragma: no cover
            filter = xpq._get_module_type_filter(op_type)
            op_type_filters.append(filter)
    for op_name, config in op_name_configs.items():
        if config.act_dtype == "fp16":  # pragma: no cover
            filter = xpq._get_module_name_filter(op_name)
            op_name_filters.append(filter)
    node_set_from_user_config = set()
    all_filters = op_type_filters + op_name_filters
    for node in gm.graph.nodes:  # pragma: no cover
        if any([filter(node) for filter in all_filters]):
            node_set_from_user_config.add(node)
    return node_set_from_user_config


def get_half_precision_node_set(gm, config):
    """Retrieves a set of nodes from the given graph model (gm) that are candidates for conversion to half precision.

    The result is the intersection between `unquantized_node_set` and `node_set_from_user_config`.

    Args:
        gm (GraphModel): The graph model to search for nodes.
        config (dict): User configuration for node candidate set.

    Returns:
        set: A set of nodes that are candidates for conversion to half precision.
    """
    # TODO: implement it, current return all unquantized_node_set

    node_set_from_user_config = _parse_node_candidate_set_from_user_config(config, gm)
    unquantized_node_set = get_unquantized_node_set(gm)
    possible_node_set = unquantized_node_set.intersection(node_set_from_user_config)
    half_precision_node_set = set()
    for node in possible_node_set:
        if node.target in SUPPORTED_OPERATORS:
            half_precision_node_set.add(node)
    utils.logger.info(f"Found {len(half_precision_node_set)} nodes to convert to half precision.")
    return half_precision_node_set
