#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
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

"""Tuning sampler."""

import copy
from collections import OrderedDict, defaultdict, deque
from itertools import product
from typing import Any, Dict, List, Tuple, Union

from ...utils import logger
from ..utils.constant import WOQ_TUNING_ALGOS
from .tuning_space import TuningSpace, pattern_to_internal, quant_mode_from_pattern
from .tuning_structs import OpTuningConfig
from .utility import ClassRegister

TUNING_ITEM_PRIORITY = [
    ("activation", "scheme"),
    ("activation", "algorithm"),
    ("activation", "granularity"),
    ("activation", "compute_dtype"),
    ("weight", "scheme"),
    ("weight", "algorithm"),
    ("weight", "granularity"),
]


tuning_sampler_dict = ClassRegister()


class TuningOrder:
    """Not displayed in API Docs."""

    def __init__(self):
        """For future use."""
        pass


class TuningSampler:
    """Not displayed in API Docs.

    Basic class of tuning sampler.
    """

    def __init__(
        self,
        tuning_space: TuningSpace,
        tuning_order_lst: List[TuningOrder],
        initial_op_tuning_cfg: Dict,
        kwargs: Dict = {},
    ):
        """Init tuning sampler.

        Args:
            tuning_space: The tuning space.
            tuning_order_lst: The traverse orders.
            initial_op_tuning_cfg: The initialized tuning config.
            kwargs: other args.
        """
        self.tuning_space = tuning_space
        self.tuning_order_lst = tuning_order_lst
        self.initial_op_tuning_cfg = initial_op_tuning_cfg
        self.queue = deque()
        # (op_name, op_type): [full_path1, full_path2,...]
        self.op_complete_path = {}

    def __iter__(self, tune_cfg=None):
        """Interface for generate the next tuning config."""
        pass

    def _set_dtype(self, op_name_type, config_args):
        has_weight = op_name_type in self.tuning_space.ops_attr["weight"]
        path = self.op_complete_path[op_name_type].get("activation", None)
        config_args["activation_dtype"] = self.tuning_space.ops_data_type[op_name_type][path]
        if has_weight:
            path = self.op_complete_path[op_name_type].get("weight", None)
            config_args["weight_dtype"] = self.tuning_space.ops_data_type[op_name_type][path]


class ModelWiseTuningSampler(TuningSampler):
    """Not displayed in API Docs."""

    def __init__(
        self,
        tuning_space: TuningSpace,
        tuning_items_priority: List[str],
        tuning_order_lst: List[TuningOrder],
        op_dtype_dict: Dict[tuple, str],
        initial_op_tuning_cfg: Dict[tuple, OpTuningConfig],
    ):
        """Model type wise tuning sampler.

        step1. create a default tuning config for each op
        step2. collect all tuning items and options, and build the model-wise traverse order
        step3. yield the tuning item with option one by one, query the existence of tuning item
               and specific option for one op if exist, use the default tuning config if not exist

        Args:
            tuning_space: Tuning space.
            tuning_items_priority: The priority to traverse the tuning items.
            tuning_order_lst: The tuning orders.
            op_dtype_dict: The (op name, op type) and its target data type.
            initial_op_tuning_cfg: The initial tuning config.
        """
        super().__init__(tuning_space, tuning_order_lst, initial_op_tuning_cfg)

        self.op_dtype_dict = op_dtype_dict
        self.tuning_space = tuning_space
        self.default_op_config = {}
        tuning_items = defaultdict(set)  # item name: options
        for op_name_type, quant_mode in op_dtype_dict.items():
            full_path = self.tuning_space.get_op_default_path_by_pattern(op_name_type, quant_mode)
            self.op_complete_path[op_name_type] = copy.deepcopy(full_path)
            # step1, set the default config for each op
            self.default_op_config[op_name_type] = tuning_space.get_default_config(op_name_type, quant_mode)
            if quant_mode[0] == "precision":
                continue
            mode_items = copy.deepcopy(full_path)  # TODO refactor the initialization method
            # step2, collect all tuning items and their options
            for att in mode_items:
                if att not in full_path:
                    continue
                quant_mode_item = self.tuning_space.query_quant_mode_item_by_full_path(op_name_type, full_path[att])
                for tuning_item in quant_mode_item.options:
                    tuning_items[tuning_item.name] = tuning_items[tuning_item.name].union(tuning_item.options)
        self.tuning_items = tuning_items

    def __iter__(self):
        """Yield the next tuning config.

        Yields:
            The next tuning config.
        """
        keys = self.tuning_items.keys()
        for vals in product(*self.tuning_items.values()):
            # traverse all possible combinations by model-wise level
            tune_cfg = copy.deepcopy(self.initial_op_tuning_cfg)
            for op_name_type, quant_mode in self.op_dtype_dict.items():
                if quant_mode[0] == "precision":
                    continue
                all_exist_flag = True
                for method_name, method_val in zip(keys, vals):
                    full_path = self.op_complete_path[op_name_type]
                    if method_name[0] not in full_path:
                        continue
                    if not self.tuning_space.query_item_option(
                        op_name_type, full_path[method_name[0]], method_name, method_val
                    ):
                        all_exist_flag = False
                        tune_cfg[op_name_type] = self.default_op_config[op_name_type]
                        break
                if all_exist_flag:
                    config_args = dict(zip(keys, vals))
                    self._set_dtype(op_name_type, config_args)
                    internal_pattern = pattern_to_internal(quant_mode)
                    quant_mode = quant_mode_from_pattern(internal_pattern)
                    tune_cfg[op_name_type] = OpTuningConfig(
                        op_name_type[0], op_name_type[1], quant_mode, self.tuning_space, kwargs=config_args
                    )
            yield tune_cfg


class OpTypeWiseTuningSampler(TuningSampler):
    """Not displayed in API Docs."""

    def __init__(
        self,
        tuning_space: TuningSpace,
        tuning_items_priority: List[str],
        tuning_order_lst: List[TuningOrder],
        op_dtype_dict: Dict[tuple, str],
        initial_op_tuning_cfg: Dict[tuple, OpTuningConfig],
    ):
        """Op type wise tuning sampler.

        Args:
            tuning_space: Tuning space.
            tuning_items_priority: The priority to traverse the tuning items.
            tuning_order_lst: The tuning orders.
            op_dtype_dict: The (op name, op type) and its target data type.
            initial_op_tuning_cfg: The initial tuning config.
        """
        super().__init__(tuning_space, tuning_order_lst, initial_op_tuning_cfg)
        tuning_items_priority = TUNING_ITEM_PRIORITY
        # (op_type, quant_mode) : {tuning_item_name : [option1, option2]}
        #  {('activation', 'scheme'): ['sym', 'sym'], ('activation', 'algorithm'): ['minmax', 'kl', 'minmax', 'kl']}

        self.optype_quant_mode_option = {}
        self.optype_quant_mode_items_name = defaultdict(list)
        self.op_type_quant_mode_wise_combination = {}
        self.op_dtype_dict = op_dtype_dict
        self.default_op_config = {}

        for op_name_type, quant_mode in op_dtype_dict.items():
            full_path = self.tuning_space.get_op_default_path_by_pattern(op_name_type, quant_mode)
            self.op_complete_path[op_name_type] = copy.deepcopy(full_path)
            self.default_op_config[op_name_type] = self.tuning_space.get_default_config(op_name_type, quant_mode)
            op_name, op_type = op_name_type
            if quant_mode[0] == "precision":
                continue
            mode_items = copy.deepcopy(full_path)  # TODO refactor the initialization method
            op_type_quant_mode = (op_type, quant_mode)
            filtered_tuning_items = []
            for item_name in tuning_items_priority:
                att, method_name = item_name
                if att not in mode_items:
                    continue
                quant_mode_item = self.tuning_space.query_quant_mode_item_by_full_path(op_name_type, full_path[att])
                item = quant_mode_item.get_option_by_name(item_name)
                if item:
                    if op_type_quant_mode not in self.optype_quant_mode_option:
                        self.optype_quant_mode_option[op_type_quant_mode] = defaultdict(list)
                    self.optype_quant_mode_option[op_type_quant_mode][item_name] += item.options
                    filtered_tuning_items.append(item)
            self.optype_quant_mode_items_name[op_type_quant_mode] = filtered_tuning_items

        for op_type_quant_mode, val in self.optype_quant_mode_option.items():
            options_lst = []
            # remove the duplicate options
            for _, item_options in val.items():
                seen = set()
                filter_options = [option for option in item_options if not (option in seen or seen.add(option))]
                options_lst.append(filter_options)
            op_type_quant_mode_vals = product(*options_lst)
            self.op_type_quant_mode_wise_combination[op_type_quant_mode] = op_type_quant_mode_vals

    def __iter__(self):
        """Yield the next tuning config.

        Yields:
            The next tuning config.
        """
        new_tune_cfg = copy.deepcopy(self.initial_op_tuning_cfg)
        for options_lst in product(*self.op_type_quant_mode_wise_combination.values()):
            for index, op_type_quant_mode in enumerate(self.op_type_quant_mode_wise_combination.keys()):
                for op_name_type, quant_mode in self.op_dtype_dict.items():
                    if op_name_type[1] == op_type_quant_mode[0] and quant_mode == op_type_quant_mode[1]:
                        op_tuning_items = [item.name for item in self.optype_quant_mode_items_name[op_type_quant_mode]]
                        op_tuning_item_vals = options_lst[index]
                        all_exist_flag = True
                        for method_name, method_val in zip(op_tuning_items, op_tuning_item_vals):
                            full_path = self.op_complete_path[op_name_type]
                            if not self.tuning_space.query_item_option(
                                op_name_type, full_path[method_name[0]], method_name, method_val
                            ):
                                all_exist_flag = False
                                op_tuning_config = self.default_op_config[op_name_type]
                                break
                        if all_exist_flag:
                            config_args = dict(zip(op_tuning_items, op_tuning_item_vals))
                            self._set_dtype(op_name_type, config_args)
                            internal_pattern = pattern_to_internal(quant_mode)
                            quant_mode = quant_mode_from_pattern(internal_pattern)
                            op_tuning_config = OpTuningConfig(
                                op_name_type[0], op_name_type[1], quant_mode, self.tuning_space, kwargs=config_args
                            )
                        new_tune_cfg.update({op_name_type: op_tuning_config})
            yield new_tune_cfg


class OpWiseTuningSampler(TuningSampler):
    """Not displayed in API Docs."""

    def __init__(
        self,
        tuning_space: TuningSpace,
        tuning_items_priority: List[str],
        tuning_order_lst: List[TuningOrder],
        op_dtype_dict: Dict[tuple, str],
        initial_op_tuning_cfg: Dict,
    ):
        """Op wise tuning config sampler.

        Args:
            tuning_space: Tuning space.
            tuning_items_priority: The priority to traverse the tuning items.
            tuning_order_lst: The tuning orders.
            op_dtype_dict: The (op name, op type) and its target data type.
            initial_op_tuning_cfg: The initial tuning config.
        """
        super().__init__(tuning_space, tuning_order_lst, initial_op_tuning_cfg)
        tuning_items_priority = TUNING_ITEM_PRIORITY
        # query the combination of tuning items with according to the tuning items priority
        self.op_dtype_dict = op_dtype_dict
        self.op_options_combination = OrderedDict()
        self.op_tuning_items = {}
        for op_name_type, op_quant_mode in op_dtype_dict.items():
            full_path = self.tuning_space.get_op_default_path_by_pattern(op_name_type, op_quant_mode)
            self.op_complete_path[op_name_type] = copy.deepcopy(full_path)
            mode_items = copy.deepcopy(full_path)
            internal_pattern = pattern_to_internal(op_quant_mode)
            op_quant_mode = quant_mode_from_pattern(internal_pattern)
            if internal_pattern[0] == "precision":
                continue
            filtered_tuning_items = []
            for item_name in tuning_items_priority:
                att, method_name = item_name
                if att not in mode_items:
                    continue
                quant_mode_item = self.tuning_space.query_quant_mode_item_by_full_path(op_name_type, full_path[att])
                item = quant_mode_item.get_option_by_name(item_name)
                if item:
                    filtered_tuning_items.append(item)
            self.op_tuning_items[op_name_type] = filtered_tuning_items
            op_options_lst = product(*[item.options for item in filtered_tuning_items])
            self.op_options_combination[op_name_type] = op_options_lst

    def __iter__(self):
        """Yield the next tuning config.

        Yields:
            The next tuning config.
        """
        new_tune_cfg = copy.deepcopy(self.initial_op_tuning_cfg)
        for op_options_lst in product(*self.op_options_combination.values()):
            for index, op_name_type in enumerate(self.op_options_combination.keys()):
                op_quant_mode = self.op_dtype_dict[op_name_type]
                op_tuning_items = [item.name for item in self.op_tuning_items[op_name_type]]
                op_tuning_item_vals = op_options_lst[index]
                config_args = dict(zip(op_tuning_items, op_tuning_item_vals))
                self._set_dtype(op_name_type, config_args)
                internal_pattern = pattern_to_internal(op_quant_mode)
                quant_mode = quant_mode_from_pattern(internal_pattern)
                op_tuning_config = OpTuningConfig(
                    op_name_type[0], op_name_type[1], quant_mode, self.tuning_space, kwargs=config_args
                )
                new_tune_cfg.update({op_name_type: op_tuning_config})
            yield new_tune_cfg

    def get_opwise_candidate(self):
        """Collect all op-wise setting.

        Returns:
            op_wise_configs: all op-wise setting.
        """
        op_wise_configs = OrderedDict()
        for op_name_type, op_quant_mode in self.op_dtype_dict.items():
            # For static/dynamic/fp32/bf16
            internal_pattern = pattern_to_internal(op_quant_mode)
            quant_mode = quant_mode_from_pattern(internal_pattern)
            full_path = self.tuning_space.get_op_default_path_by_pattern(op_name_type, op_quant_mode)
            self.op_complete_path[op_name_type] = copy.deepcopy(full_path)
            op_wise_configs[op_name_type] = []
            # For precision
            if internal_pattern[0] == "precision":
                config_args = {}
                self._set_dtype(op_name_type, config_args)
                op_tuning_config = OpTuningConfig(
                    op_name_type[0], op_name_type[1], quant_mode, self.tuning_space, kwargs=config_args
                )
                op_wise_configs[op_name_type].append(op_tuning_config)
                continue
            # For quantization
            op_tuning_items = [item.name for item in self.op_tuning_items.get(op_name_type, [])]
            op_options = self.op_options_combination[op_name_type]

            for op_tuning_item_vals in op_options:
                config_args = dict(zip(op_tuning_items, op_tuning_item_vals))
                self._set_dtype(op_name_type, config_args)
                op_tuning_config = OpTuningConfig(
                    op_name_type[0], op_name_type[1], quant_mode, self.tuning_space, kwargs=config_args
                )
                op_wise_configs[op_name_type].append(op_tuning_config)
        return op_wise_configs


class FallbackTuningSampler(TuningSampler):
    """Not displayed in API Docs."""

    def __init__(
        self,
        tuning_space: TuningSpace,
        tuning_order_lst: List[TuningOrder],
        initial_op_tuning_cfg: Dict[Tuple, Any],
        op_dtypes: Dict[Union[Tuple, Tuple[Tuple]], str],
        accumulate: bool,
        skip_first: bool = True,
    ):
        """Sampler for generate the tuning config of fallback stage.

        Args:
            tuning_space: Tuning space.
            tuning_order_lst: The tuning orders.
            initial_op_tuning_cfg: The initial tuning config.
            op_dtypes: The (op name, op type) and its target data type.
            accumulate: Fallback accumulated or not.
            skip_first: Skip fallback the first op or not. Defaults to True.
        """
        super().__init__(tuning_space, tuning_order_lst, initial_op_tuning_cfg)
        self.op_dtypes = op_dtypes
        self.accumulate = accumulate
        self.skip_first = skip_first

    def __iter__(self):
        """Yield the next tuning config.

        Yields:
            The next tuning config.
        """
        new_tune_cfg = copy.deepcopy(self.initial_op_tuning_cfg)
        skip_first = self.skip_first
        for op_name_type, target_dtype in self.op_dtypes.items():
            # Only support fallback to lower precision.
            if not self.accumulate:
                new_tune_cfg = copy.deepcopy(self.initial_op_tuning_cfg)
            op_name_type_lst = (
                [op_name_type] if len(op_name_type) != 1 and isinstance(op_name_type[1], str) else op_name_type
            )
            for op_name_type in op_name_type_lst:
                full_path = self.tuning_space.get_op_default_path_by_pattern(op_name_type, target_dtype)
                self.op_complete_path[op_name_type] = copy.deepcopy(full_path)
                config_args = {}
                self._set_dtype(op_name_type, config_args)
                internal_pattern = pattern_to_internal(target_dtype)
                quant_mode = quant_mode_from_pattern(internal_pattern)
                new_op_config = OpTuningConfig(
                    op_name_type[0], op_name_type[1], quant_mode, self.tuning_space, kwargs=config_args
                )

                new_tune_cfg.update({op_name_type: new_op_config})
            if self.accumulate and skip_first:  # skip the first one
                skip_first = False
                continue
            logger.info(f"fallback {op_name_type_lst} to {target_dtype}")
            yield new_tune_cfg  # need to skip the first one


class LowerBitsSampler(TuningSampler):
    """Not displayed in API Docs."""

    def __init__(
        self,
        tuning_space: TuningSpace,
        tuning_order_lst: List[TuningOrder],
        initial_op_tuning_cfg: Dict[tuple, Any],
        op_dtypes: Dict[str, str],
        accumulate: bool,
        skip_first: bool = True,
    ):
        """Generate tuning config with lower bits.

        Args:
            tuning_space: Tuning space.
            tuning_order_lst: The tuning orders.
            initial_op_tuning_cfg: The initial tuning config.
            op_dtypes: The (op name, op type) and its target data type.
            accumulate: Fallback accumulated or not.
            skip_first: Skip fallback the first op or not. Defaults to True.
        """
        super().__init__(tuning_space, tuning_order_lst, initial_op_tuning_cfg)
        self.op_dtypes = op_dtypes
        self.accumulate = accumulate
        self.skip_first = skip_first

    def __iter__(self):
        """Yield the next tuning config.

        Yields:
            The next tuning config.
        """
        new_tune_cfg = copy.deepcopy(self.initial_op_tuning_cfg)
        skip_first = self.skip_first
        for op_name_type, target_dtype in self.op_dtypes.items():
            # Only support fallback to lower precision.
            if not self.accumulate:
                new_tune_cfg = copy.deepcopy(self.initial_op_tuning_cfg)
            full_path = self.tuning_space.get_op_default_path_by_quant_bits(op_name_type, target_dtype)
            self.op_complete_path[op_name_type] = copy.deepcopy(full_path)
            config_args = {}
            self._set_dtype(op_name_type, config_args)
            new_op_config = _get_default_config_by_path(op_name_type, self.tuning_space, full_path)
            new_tune_cfg.update({op_name_type: new_op_config})
            if self.accumulate and skip_first:  # skip the first one
                skip_first = False
                continue
            logger.debug(f"Quantize {op_name_type} to {target_dtype}")
            yield new_tune_cfg  # need to skip the first one


def _get_default_config_by_path(op_name_type, tuning_space, full_path):
    """Get default config according to path."""
    from .constant import TUNING_ITEMS_LST

    has_weight = op_name_type in tuning_space.ops_attr["weight"]
    config_args = {}
    att_lst = ["activation", "weight"] if has_weight else ["activation"]
    for att in att_lst:
        att_full_path = tuning_space.get_default_full_path(op_name_type, full_path[att])
        config_args[att + "_dtype"] = tuning_space.ops_data_type[op_name_type].get(att_full_path, None)
        mode_item = tuning_space.get_item_by_path((op_name_type, *att_full_path))
        if mode_item:
            method_args = {
                method_item.name: method_item.options[0]
                for method_item in mode_item.options
                if method_item.name in TUNING_ITEMS_LST
            }
            config_args.update(method_args)
    quant_mode = full_path["weight"][0]
    # set the first option as the default for each tuning item
    op_tuning_config = OpTuningConfig(op_name_type[0], op_name_type[1], quant_mode, tuning_space, kwargs=config_args)
    return op_tuning_config


class BlockFallbackTuningSampler(TuningSampler):
    """Not displayed in API Docs."""

    def __init__(
        self,
        tuning_space: TuningSpace,
        tuning_order_lst: List[TuningOrder],
        initial_op_tuning_cfg: Dict[tuple, Any],
        op_block_lst: List[List[tuple]],
        accumulate: bool,
        target_dtype: str,
    ):
        """Sampler for generate the tuning config of fallback stage.

        Args:
            tuning_space (TuningSpace): Tuning space.
            tuning_order_lst (List[TuningOrder]): The tuning orders.
            initial_op_tuning_cfg (Dict[tuple, Any]): The initial tuning config.
            op_block_lst (List[List[tuple]]): The block of op_list,
                [[(op name, op type), (op name, op type), ...], op_list2, ...].
            accumulate (bool): Fallback accumulated or not.
            target_dtype (str): Skip fallback the first op or not. Defaults to True.
        """
        super().__init__(tuning_space, tuning_order_lst, initial_op_tuning_cfg)
        self.op_block_lst = op_block_lst
        self.accumulate = accumulate
        self.target_dtype = target_dtype

    def __iter__(self):
        """Yield the next tuning config.

        Yields:
            The next tuning config.
        """
        new_tune_cfg = copy.deepcopy(self.initial_op_tuning_cfg)
        for op_block in self.op_block_lst:
            # Only support fallback to lower precision.
            if not self.accumulate:
                new_tune_cfg = copy.deepcopy(self.initial_op_tuning_cfg)
            logger.debug(f"[BlockFallbackTuningSampler] op_block: {op_block}")
            for op_name_type in op_block:
                full_path = self.tuning_space.get_op_default_path_by_pattern(op_name_type, self.target_dtype)
                self.op_complete_path[op_name_type] = copy.deepcopy(full_path)
                config_args = {}
                self._set_dtype(op_name_type, config_args)
                internal_pattern = pattern_to_internal(self.target_dtype)
                quant_mode = quant_mode_from_pattern(internal_pattern)
                new_op_config = OpTuningConfig(
                    op_name_type[0], op_name_type[1], quant_mode, self.tuning_space, kwargs=config_args
                )

                new_tune_cfg.update({op_name_type: new_op_config})
                logger.debug(f"[BlockFallbackTuningSampler] updated_tuning_cfg {op_name_type}: {new_op_config}")
                logger.debug(f"[BlockFallbackTuningSampler] fallback {op_name_type} to {self.target_dtype}")
            yield new_tune_cfg


@tuning_sampler_dict("smooth_quant")
class SmoothQuantSampler(TuningSampler):
    """Not displayed in API Docs."""

    def __init__(
        self,
        tuning_space: TuningSpace,
        tuning_order_lst: List[TuningOrder],
        initial_op_tuning_cfg: Dict,
        alpha_list: List[float],
        kwargs: Dict = {},
    ):
        """Init tuning sampler.

        Args:
            tuning_space: The tuning space.
            tuning_order_lst: The traverse orders.
            initial_op_tuning_cfg: The initialized tuning config.
            alpha_list: smooth quant alpha list.
            kwargs: other args.
        """
        super().__init__(tuning_space, tuning_order_lst, initial_op_tuning_cfg)
        self.sq_alpha_list = alpha_list

    def __iter__(self):
        """Yield the next tuning config.

        Yields:
            The next tuning config.
        """
        new_tune_cfg = copy.deepcopy(self.initial_op_tuning_cfg)
        logger.debug(f"[STRATEGY] smooth quant alpha list: {self.sq_alpha_list}")
        for alpha in self.sq_alpha_list:
            recipe_cfgs = new_tune_cfg.setdefault("recipe_cfgs", {})
            recipe_cfgs["smooth_quant"] = True
            recipe_cfgs["smooth_quant_args"] = {"alpha": alpha}
            logger.debug(f"[STRATEGY] set smooth quant alpha with: {alpha:.4f}")
            yield new_tune_cfg


@tuning_sampler_dict("woq_algorithm")
class WeightOnlyQuantSampler(TuningSampler):
    """Not displayed in API Docs."""

    def __init__(
        self,
        tuning_space: TuningSpace,
        tuning_order_lst: List[TuningOrder],
        initial_op_tuning_cfg: Dict,
    ):
        """Init tuning sampler.

        Args:
            tuning_space: The tuning space.
            tuning_order_lst: The traverse orders.
            initial_op_tuning_cfg: The initialized tuning config.
        """
        super().__init__(tuning_space, tuning_order_lst, initial_op_tuning_cfg)

    def __iter__(self):
        """Yield the next tuning config.

        Yields:
            The next tuning config.
        """
        new_tune_cfg = copy.deepcopy(self.initial_op_tuning_cfg)
        for algo in WOQ_TUNING_ALGOS.keys():
            new_tune_cfg["woq_tuning_cfg"] = algo
            yield new_tune_cfg
