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
"""The basic tuning strategy."""
from collections import OrderedDict
from copy import deepcopy

from deprecated import deprecated

from ...utils import logger
from .strategy import TuneStrategy, strategy_registry
from .utils.constant import TUNING_ITEMS_LST
from .utils.tuning_sampler import FallbackTuningSampler, OpTypeWiseTuningSampler
from .utils.tuning_structs import OpTuningConfig


@deprecated(version="2.0")
@strategy_registry
class BasicTuneStrategy(TuneStrategy):
    """The basic tuning strategy.

    There are three stages executed by Basic strategy sequentially,
    and the tuning process ends once the condition meets the exit policy.
    """

    def next_tune_cfg(self):
        """Generate and yield the next tuning config with below order.

            1. OP Type Wise Tuning: tries to quantize the OPs as many as possible
                and traverse all OP type wise tuning configs
            2. Fallback OP One by One: it performs high-precision OP (FP32, BF16 ...)
                fallbacks one by one based on the tuning config with the best result
                in the previous stage, and records the impact of each OP.
            3. Fallback Multiple OPs Accumulated: first sorted the OPs list
                according to the impact score in stage II, and tries to incrementally
                fallback multiple OPs to high precision according to the sorted OP list.

        Returns:
            tune_config (dict): A dict containing the tuning configuration for quantization.
        """
        tuning_space = self.tuning_space
        calib_sampling_size_lst = tuning_space.root_item.get_option_by_name("calib_sampling_size").options
        for calib_sampling_size in calib_sampling_size_lst:
            # Initialize the tuning config for each op according to the quantization approach.
            op_item_dtype_dict, quant_mode_wise_items, initial_op_tuning_cfg = self.initial_tuning_cfg()
            # Optype-wise tuning tuning items: the algorithm/scheme/granularity of activation(weight)
            early_stop_tuning = False
            stage1_cnt = 0
            quant_ops = quant_mode_wise_items.get("static", [])
            quant_ops += quant_mode_wise_items.get("dynamic", [])
            stage1_max = 1e9  # TODO set a more appropriate value
            op_type_wise_tuning_sampler = OpTypeWiseTuningSampler(
                tuning_space, [], [], op_item_dtype_dict, initial_op_tuning_cfg
            )
            for index, op_tuning_cfg in enumerate(op_type_wise_tuning_sampler):
                logger.debug(f"[OP TYPE WISE STAGE], Trial {index + 1}")
                op_tuning_cfg["calib_sampling_size"] = calib_sampling_size
                # Apply all recipes, if not got the qmodel that meet the requirements, discard it.
                if index == 1 and not self.applied_all_recipes_flag:
                    logger.info("Apply all recipes.")
                    self.applied_all_recipes_flag = True
                    yield self.apply_all_tuning_recipes(deepcopy(self.cur_best_tuning_cfg))
                stage1_cnt += 1
                if early_stop_tuning and stage1_cnt > stage1_max:
                    logger.info("Early stopping the stage 1.")
                    break
                yield op_tuning_cfg

            # Apply all recipes, if not got the qmodel that meet the requirements, discard it.
            if stage1_cnt == 1 and not self.applied_all_recipes_flag:
                logger.info("Apply all recipes.")
                self.applied_all_recipes_flag = True
                yield self.apply_all_tuning_recipes(deepcopy(self.cur_best_tuning_cfg))

            # Fallback the ops supported both static and dynamic from static to dynamic
            # Tuning items: None
            if self.cfg.quantization.approach == "post_training_auto_quant":
                static_dynamic_items = [
                    item
                    for item in tuning_space.query_items_by_quant_mode("static")
                    if item in tuning_space.query_items_by_quant_mode("dynamic")
                ]
                if static_dynamic_items:
                    logger.info("Fallback all ops that support both dynamic and static to dynamic.")
                else:
                    logger.info("Non ops that support both dynamic")

                new_op_tuning_cfg = deepcopy(self.cur_best_tuning_cfg)
                for item in static_dynamic_items:
                    new_op_tuning_cfg[item.name] = self._initial_dynamic_cfg_based_on_static_cfg(
                        new_op_tuning_cfg[item.name]
                    )
                new_op_tuning_cfg["calib_sampling_size"] = calib_sampling_size
                yield new_op_tuning_cfg

            logger.info("Apply recipe one by one.")
            for tune_cfg in self.apply_recipe_one_by_one(deepcopy(self.cur_best_tuning_cfg)):
                yield tune_cfg
            best_op_tuning_cfg_stage1 = deepcopy(self.cur_best_tuning_cfg)

            # Fallback
            for target_dtype in ["bf16", "fp32"]:
                target_type_lst = set(tuning_space.query_items_by_quant_mode(target_dtype))
                fallback_items_lst = [item for item in quant_ops if item in target_type_lst]
                if fallback_items_lst:
                    logger.info(f"Start to fallback op to {target_dtype} one by one.")
                    self._fallback_started()
                fallback_items_name_lst = [item.name for item in fallback_items_lst][::-1]  # from bottom to up
                op_dtypes = OrderedDict(zip(fallback_items_name_lst, [target_dtype] * len(fallback_items_name_lst)))
                initial_op_tuning_cfg = deepcopy(best_op_tuning_cfg_stage1)
                fallback_sampler = FallbackTuningSampler(
                    tuning_space,
                    tuning_order_lst=[],
                    initial_op_tuning_cfg=initial_op_tuning_cfg,
                    op_dtypes=op_dtypes,
                    accumulate=False,
                )
                op_fallback_acc_impact = OrderedDict()
                for op_index, op_tuning_cfg in enumerate(fallback_sampler):
                    op_tuning_cfg["calib_sampling_size"] = calib_sampling_size
                    yield op_tuning_cfg
                    acc, _ = self.last_tune_result
                    op_fallback_acc_impact[fallback_items_name_lst[op_index]] = acc

                # Fallback OPs accumulated according to the order in the previous stage
                if len(op_fallback_acc_impact) > 0:
                    ordered_ops = sorted(
                        op_fallback_acc_impact.keys(),
                        key=lambda key: op_fallback_acc_impact[key],
                        reverse=self.higher_is_better,
                    )
                    op_dtypes = OrderedDict(zip(ordered_ops, [target_dtype] * len(fallback_items_name_lst)))
                    logger.info(f"Start to accumulate fallback to {target_dtype}.")
                    initial_op_tuning_cfg = deepcopy(best_op_tuning_cfg_stage1)
                    fallback_sampler = FallbackTuningSampler(
                        tuning_space,
                        tuning_order_lst=[],
                        initial_op_tuning_cfg=initial_op_tuning_cfg,
                        op_dtypes=op_dtypes,
                        accumulate=True,
                    )
                    for op_tuning_cfg in fallback_sampler:
                        op_tuning_cfg["calib_sampling_size"] = calib_sampling_size
                        yield op_tuning_cfg

    def _initial_dynamic_cfg_based_on_static_cfg(self, op_static_cfg: OpTuningConfig):
        op_state = op_static_cfg.get_state()
        op_name = op_static_cfg.op_name
        op_type = op_static_cfg.op_type
        op_name_type = (op_name, op_type)
        op_quant_mode = "dynamic"
        tuning_space = self.tuning_space
        dynamic_state = {}
        for att in ["weight", "activation"]:
            if att not in op_state:
                continue
            # Add dtype
            full_path = self.tuning_space.get_op_default_path_by_pattern(op_name_type, op_quant_mode)
            dynamic_state[att + "_dtype"] = self.tuning_space.ops_data_type[op_name_type][full_path[att]]
            for method_name, method_val in op_state[att].items():
                att_and_method_name = (att, method_name)
                if att_and_method_name not in TUNING_ITEMS_LST:
                    continue
                if tuning_space.query_item_option(op_name_type, full_path[att], att_and_method_name, method_val):
                    dynamic_state[att_and_method_name] = method_val
                else:
                    quant_mode_item = tuning_space.get_item_by_path((op_name_type, *full_path[att]))
                    if quant_mode_item and quant_mode_item.get_option_by_name(att_and_method_name):
                        tuning_item = quant_mode_item.get_option_by_name(att_and_method_name)
                        dynamic_state[att_and_method_name] = tuning_item.options[0] if tuning_item else None
        return OpTuningConfig(op_name, op_type, op_quant_mode, tuning_space, kwargs=dynamic_state)
