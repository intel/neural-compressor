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

from ..utils import logger
from .strategy import TuneStrategy, strategy_registry
from .utils.constant import LOWER_BIT_LIST, PRECISION_LIST
from .utils.tuning_sampler import (
    BlockFallbackTuningSampler,
    FallbackTuningSampler,
    LowerBitsSampler,
    OpTypeWiseTuningSampler,
)
from .utils.tuning_structs import OpTuningConfig


@strategy_registry
class BasicTuneStrategy(TuneStrategy):
    """The basic tuning strategy.

    There are three stages executed by Basic strategy sequentially,
    and the tuning process ends once the condition meets the exit policy.
    """

    def distributed_next_tune_cfg_lst(self, comm):
        """Generate and yield the next tuning config list with below order.

            1. OP Type Wise Tuning
            2. Fallback OPs Block by Block
            3. Fallback OP One by One
            4. Fallback Multiple OPs Accumulated

        Yields:
            tuning_config_list (list): A list containing dicts of the tuning configuration for quantization.
        """
        from copy import deepcopy

        tuning_space = self.tuning_space
        calib_sampling_size_lst = tuning_space.root_item.get_option_by_name("calib_sampling_size").options
        rank = comm.Get_rank()
        for calib_sampling_size in calib_sampling_size_lst:
            # Initialize the tuning config for each op according to the quantization approach
            op_item_dtype_dict, quant_mode_wise_items, initial_op_tuning_cfg = self.initial_tuning_cfg()
            # Optype-wise tuning tuning items: the algorithm/scheme/granularity of activation(weight)
            early_stop_tuning = False
            stage1_cnt = 0
            quant_ops = quant_mode_wise_items["static"] if "static" in quant_mode_wise_items else []
            quant_ops += quant_mode_wise_items["dynamic"] if "dynamic" in quant_mode_wise_items else []
            stage1_max = 1e9  # TODO set a more appropriate value
            if not self.cur_best_tuning_cfg:
                self.cur_best_tuning_cfg = deepcopy(initial_op_tuning_cfg)

            # try to tune sq alpha
            op_tuning_cfg_lst_stage_sq = []
            if self._should_tuning_sq_alpha(self.config.recipes):
                for tune_cfg in self.tuning_sq_alpha(
                    tuning_space, deepcopy(self.cur_best_tuning_cfg), self.config.recipes
                ):
                    op_tuning_cfg_lst_stage_sq.append(tune_cfg)
            yield op_tuning_cfg_lst_stage_sq

            # op type-wise tuning
            op_type_wise_tuning_sampler = OpTypeWiseTuningSampler(
                tuning_space, [], [], op_item_dtype_dict, initial_op_tuning_cfg
            )
            # stage 1: yield op_tune_cfg_lst
            op_tuning_cfg_lst_stage_1 = []
            for op_tuning_cfg in op_type_wise_tuning_sampler:
                stage1_cnt += 1
                if early_stop_tuning and stage1_cnt > stage1_max:
                    logger.info("Early stopping the stage 1.")
                    break
                op_tuning_cfg["calib_sampling_size"] = calib_sampling_size
                op_tuning_cfg_lst_stage_1.append(deepcopy(op_tuning_cfg))
            logger.info("yield op_tuning_cfg_lst_stage_1 with length {}".format(len(op_tuning_cfg_lst_stage_1)))
            yield op_tuning_cfg_lst_stage_1

            # Coordinate: only master knows cur best tune cfg
            cur_best_tuning_cfg = self.cur_best_tuning_cfg if rank == 0 else None
            if rank == 0:
                comm.bcast(cur_best_tuning_cfg, root=0)
            else:
                self.cur_best_tuning_cfg = comm.bcast(cur_best_tuning_cfg, root=0)

            # stage 2: yield new_op_tuning_cfg_lst (length of stage 1)
            # Fallback the ops supported both static and dynamic from static to dynamic
            # Tuning items: None
            if self.config.approach == "post_training_auto_quant":
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
                    new_op_tuning_cfg[item.name] = self.initial_dynamic_cfg_based_on_static_cfg(
                        new_op_tuning_cfg[item.name]
                    )
                new_op_tuning_cfg["calib_sampling_size"] = calib_sampling_size
                op_tuning_cfg_lst_stage_2 = [deepcopy(new_op_tuning_cfg)]
                logger.info("yield op_tuning_cfg_lst_stage_2 with length {}".format(len(op_tuning_cfg_lst_stage_2)))
                yield op_tuning_cfg_lst_stage_2

            # Coordinate: only master knows cur best tune cfg
            cur_best_tuning_cfg = self.cur_best_tuning_cfg if rank == 0 else None
            if rank == 0:
                comm.bcast(cur_best_tuning_cfg, root=0)
            else:
                self.cur_best_tuning_cfg = comm.bcast(cur_best_tuning_cfg, root=0)

            best_op_tuning_cfg_stage1 = deepcopy(self.cur_best_tuning_cfg)

            # Fallback
            # Fallback block after stage (1, 2) and before stage (3, 4)
            # stage 3, 4: yield op_tuning_cfg_lst
            op_tuning_cfg_lst_stage_block = []
            op_tuning_cfg_lst_stage_3 = []
            op_tuning_cfg_lst_stage_4 = []
            for target_dtype in PRECISION_LIST:
                target_type_lst = set(tuning_space.query_items_by_quant_mode(target_dtype))
                fallback_items_lst = [item for item in quant_ops if item in target_type_lst]

                # Fallback block by block
                for op_tuning_cfg in self.fallback_by_block(
                    fallback_items_lst, best_op_tuning_cfg_stage1, target_dtype, tuning_space, calib_sampling_size
                ):
                    op_tuning_cfg_lst_stage_block.append(deepcopy(op_tuning_cfg))
                logger.info(
                    "yield op_tuning_cfg_lst_stage_block with length {}".format(len(op_tuning_cfg_lst_stage_block))
                )
                yield op_tuning_cfg_lst_stage_block

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
                    # yield op_tuning_cfg
                    op_tuning_cfg_lst_stage_3.append(deepcopy(op_tuning_cfg))
                logger.info("yield op_tuning_cfg_lst_stage_3 with length {}".format(len(op_tuning_cfg_lst_stage_3)))
                yield op_tuning_cfg_lst_stage_3

                # Only master updates op_fallback_acc_impact
                if rank == 0:
                    for op_index, op_tuning_cfg in enumerate(fallback_sampler):
                        acc, _ = self.eval_results[op_index]
                        op_fallback_acc_impact[fallback_items_name_lst[op_index]] = acc

                # Coordinate: only master knows op_fallback_acc_impact
                op_fallback_acc_impact = op_fallback_acc_impact if rank == 0 else None
                if rank == 0:
                    comm.bcast(op_fallback_acc_impact, root=0)
                else:
                    op_fallback_acc_impact = comm.bcast(op_fallback_acc_impact, root=0)

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
                        # yield op_tuning_cfg
                        op_tuning_cfg_lst_stage_4.append(deepcopy(op_tuning_cfg))
                    logger.info("yield op_tuning_cfg_lst_stage_4 with length {}".format(len(op_tuning_cfg_lst_stage_4)))
                    yield op_tuning_cfg_lst_stage_4

    def fallback_by_block(
        self, fallback_items_lst, best_op_tuning_cfg_stage1, target_dtype, tuning_space, calib_sampling_size
    ):
        """Fallback ops by block.

        Step 1. block by block
        Step 2. accumulate block

        Args:
            fallback_items_lst (list): list of fallback items
            best_op_tuning_cfg_stage1 (dict): best op tuning cfg of stage1
            target_dtype (str): target dtype
            tuning_space (TuningSpace): Tuning space

        Yields:
            dict: op_tuning_cfg fall-backed by block
        """
        from copy import deepcopy

        op_block_lst = self.capability.get("block_wise", [])
        fallback_items_name_lst = [item.name for item in fallback_items_lst]
        if op_block_lst and fallback_items_name_lst:
            # Fallback block by block
            op_block_fallback_lst = []
            for op_block_index, op_block in enumerate(op_block_lst):
                matches = [item for item in op_block if item in fallback_items_name_lst]
                if matches:
                    op_block_fallback_lst.append(op_block)

            initial_op_tuning_cfg = deepcopy(best_op_tuning_cfg_stage1)
            # Fallback by accumulating blocks
            if op_block_fallback_lst:
                logger.info(f"Start to fallback op to {target_dtype} by blocks")
            block_fallback_sampler = BlockFallbackTuningSampler(
                tuning_space=tuning_space,
                tuning_order_lst=[],
                initial_op_tuning_cfg=initial_op_tuning_cfg,
                op_block_lst=op_block_fallback_lst,
                accumulate=True,
                target_dtype=target_dtype,
            )
            for op_block_index, op_tuning_cfg in enumerate(block_fallback_sampler):
                op_tuning_cfg["calib_sampling_size"] = calib_sampling_size
                yield op_tuning_cfg

    def quant_to_lower_bits(self, initial_op_tuning_cfg, calib_sampling_size):
        """Quantize ops into lower bits, such as int4.

        Args:
            initial_op_tuning_cfg: the initial tuning config
            calib_sampling_size: calibration sampling size

        Yields:
            tuning config
        """
        for quant_bit in LOWER_BIT_LIST:
            logger.info(f"Start to quantize ops into {quant_bit}")
            ops = self.tuning_space.collect_op_by_quant_bits(quant_bit)
            op_item_dtype_dict = {op.name: quant_bit for op in ops}
            lower_bits_sampler = LowerBitsSampler(
                deepcopy(self.tuning_space),
                [],
                initial_op_tuning_cfg,
                op_item_dtype_dict,
                accumulate=False,
                skip_first=True,
            )
            for tune_cfg in lower_bits_sampler:
                tune_cfg["calib_sampling_size"] = calib_sampling_size
                yield tune_cfg

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
        from copy import deepcopy

        tuning_space = self.tuning_space
        calib_sampling_size_lst = tuning_space.root_item.get_option_by_name("calib_sampling_size").options
        for calib_sampling_size in calib_sampling_size_lst:
            # Initialize the tuning config for each op according to the quantization approach.
            op_item_dtype_dict, quant_mode_wise_items, initial_op_tuning_cfg = self.initial_tuning_cfg()
            initial_op_tuning_cfg["calib_sampling_size"] = calib_sampling_size
            # Optype-wise tuning tuning items: the algorithm/scheme/granularity of activation(weight)
            early_stop_tuning = False
            stage1_cnt = 0
            quant_ops = quant_mode_wise_items.get("static", [])
            quant_ops += quant_mode_wise_items.get("dynamic", [])
            stage1_max = 1e9  # TODO set a more appropriate value
            if not self.cur_best_tuning_cfg:
                self.cur_best_tuning_cfg = deepcopy(initial_op_tuning_cfg)

            # try to tune a WeightOnlyQuant algorithm
            if self._should_tuning_woq_algo():
                for tune_cfg in self.tuning_woq_algo(tuning_space, deepcopy(self.cur_best_tuning_cfg)):
                    yield tune_cfg

            # try to tune sq alpha
            if self._should_tuning_sq_alpha(self.config.recipes):
                for tune_cfg in self.tuning_sq_alpha(
                    tuning_space, deepcopy(self.cur_best_tuning_cfg), self.config.recipes
                ):
                    yield tune_cfg

            # op type-wise tuning
            op_type_wise_tuning_sampler = OpTypeWiseTuningSampler(
                tuning_space, [], [], op_item_dtype_dict, initial_op_tuning_cfg
            )

            for index, op_tuning_cfg in enumerate(op_type_wise_tuning_sampler):
                op_tuning_cfg["calib_sampling_size"] = calib_sampling_size
                # try to quantizing ops into lower bits, such as int4,
                # if accuracy meets the requirements after first trial and max_trials > 1
                if index == 1 and self.objectives.accuracy_meet_req(deepcopy(self.last_tune_result)):
                    for op_tuning_cfg in self.quant_to_lower_bits(self.cur_best_tuning_cfg, calib_sampling_size):
                        yield op_tuning_cfg
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

            # TODO Add the lower bits quantization here
            # Apply all recipes, if not got the qmodel that meet the requirements, discard it.
            if stage1_cnt == 1 and not self.applied_all_recipes_flag:
                logger.info("Apply all recipes.")
                self.applied_all_recipes_flag = True
                yield self.apply_all_tuning_recipes(deepcopy(self.cur_best_tuning_cfg))

            # Fallback the ops supported both static and dynamic from static to dynamic
            # Tuning items: None
            if self.config.approach == "post_training_auto_quant":
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
                    new_op_tuning_cfg[item.name] = self.initial_dynamic_cfg_based_on_static_cfg(
                        new_op_tuning_cfg[item.name]
                    )
                new_op_tuning_cfg["calib_sampling_size"] = calib_sampling_size
                yield new_op_tuning_cfg

            logger.info("Apply recipe one by one.")
            for tune_cfg in self.apply_recipe_one_by_one(deepcopy(self.cur_best_tuning_cfg)):
                yield tune_cfg
            best_op_tuning_cfg_stage1 = deepcopy(self.cur_best_tuning_cfg)

            # Fallback
            for target_dtype in PRECISION_LIST:
                target_type_lst = set(tuning_space.query_items_by_quant_mode(target_dtype))
                fallback_items_lst = [item for item in quant_ops if item in target_type_lst]

                # Fallback block by block
                for op_tuning_cfg in self.fallback_by_block(
                    fallback_items_lst, best_op_tuning_cfg_stage1, target_dtype, tuning_space, calib_sampling_size
                ):
                    yield op_tuning_cfg

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
            logger.warning(
                "[Strategy] All tuning options for the current strategy have been tried.\
                If the quantized model does not seem to work well, it might be worth considering other strategies."
            )
