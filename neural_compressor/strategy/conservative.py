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
"""The conservative tuning strategy for quantization level 0."""
from collections import OrderedDict as COrderedDict
from copy import deepcopy
from typing import List, OrderedDict, Tuple

from ..utils import logger
from .strategy import TuneStrategy, strategy_registry
from .utils.tuning_space import TuningItem


@strategy_registry
class ConservativeTuneStrategy(TuneStrategy):
    """Tuning strategy with accuracy first, performance second.

    The quantization level O0 is designed for user who want to keep the accuracy
    of the model after quantization. It starts with the original(fp32) model,
    and then quantize the OPs to lower precision OP type wisely and OP wisely.
    """

    def __init__(
        self,
        model,
        conf,
        q_dataloader=None,
        q_func=None,
        eval_func=None,
        eval_dataloader=None,
        eval_metric=None,
        resume=None,
        q_hooks=None,
    ):
        """Init conservative tuning strategy.

        Args:
            model: The FP32 model specified for low precision tuning.
            conf: The Conf class instance includes all user configurations.
            q_dataloader: Data loader for calibration, mandatory for post-training quantization.  Defaults to None.
            q_func: Training function for quantization aware training. Defaults to None. Defaults to None.
            eval_func: The evaluation function provided by user. This function takes model as parameter, and
                evaluation dataset and metrics should be encapsulated in this function implementation and
                outputs a higher-is-better accuracy scalar value.
            eval_dataloader: Data loader for evaluation. Defaults to None.
            eval_metric: Metric for evaluation. Defaults to None.
            resume: The dict containing resume information. Defaults to None.
            q_hooks: The dict of training hooks, supported keys are: on_epoch_begin, on_epoch_end, on_step_begin,
                on_step_end. Their values are functions to be executed in adaptor layer.. Defaults to None.
        """
        super().__init__(
            model=model,
            conf=conf,
            q_dataloader=q_dataloader,
            q_func=q_func,
            eval_func=eval_func,
            eval_dataloader=eval_dataloader,
            eval_metric=eval_metric,
            resume=resume,
            q_hooks=q_hooks,
        )
        logger.info("*** Initialize conservative tuning")
        self.acc_meet_flag = False
        # conv: conv1d, conv2d, conv3d
        self.quant_op_type_lst = ["conv", "matmul", "bmm", "linear"]
        extend_op_type_lst = self._get_extend_op_type_lst()
        self.quant_op_type_lst += extend_op_type_lst
        res_lst = [None] * len(self.quant_op_type_lst)
        self.quant_status = {k: v for k, v in zip(self.quant_op_type_lst, res_lst)}

    def _get_extend_op_type_lst(self):
        extend_lst = []
        # add 'add' to op type list when sq is on
        if self.config.recipes.get("smooth_quant", False):
            extend_lst.append("add")
        return extend_lst

    def next_tune_cfg(self):
        """Generate and yield the next tuning config with below order.

        1. Query all quantifiable ops and save as a list of [(op_name, op_type), ...]
        2. Classify the op by its op type
        3. Add op to quant_queue according to the op type priority
        4. Go through the quant_queue and replace it with the fp32 config in tune_cfg if
        accuracy meets the requirements else continue
        5. For bf16 and fp16 operators, do the same as int8 operators.

        Returns:
            tune_config (dict): It's a dict containing the tuning configuration to run.
        """
        tuning_space = self.tuning_space
        calib_sampling_size_lst = tuning_space.root_item.get_option_by_name("calib_sampling_size").options
        calib_sampling_size = calib_sampling_size_lst[0]
        op_item_dtype_dict, quant_mode_wise_items, tune_cfg = self.initialize_tune_cfg()
        tune_cfg["calib_sampling_size"] = calib_sampling_size
        op_type_priority = self._get_op_type_priority()
        quant_items_pool = self._quant_items_pool(op_type_priority)
        self.re_quant = True
        logger.info("*** Try to convert op into lower precision to improve performance.")
        for dtype, op_items in quant_items_pool.items():
            logger.info(f"*** Start to convert op into {dtype}.")
            for op_type, items_lst in op_items.items():
                logger.info(f"*** Try to convert all {op_type} ops into {dtype}.")
                tmp_tune_cfg = deepcopy(tune_cfg)
                for item, quant_mode in items_lst:
                    op_info = item.name
                    op_config = tuning_space.get_default_config(op_info, quant_mode)
                    tmp_tune_cfg[op_info] = op_config
                yield tmp_tune_cfg
                if self.objectives.accuracy_meets():
                    self.quant_status[op_type] = dtype
                    logger.info(f"*** Convert all {op_type} ops to {dtype} and accuracy still meet the requirements")
                    tune_cfg = deepcopy(tmp_tune_cfg)
                else:
                    # tmp_tune_cfg = deepcopy(tune_cfg)
                    self.quant_status[op_type] = "fp32"
                    logger.info(f"*** Convert all {op_type} ops to {dtype} but accuracy not meet the requirements")
                logger.info(f"***Current result {self.quant_status.items()}")
        logger.info("*** Ending tuning process due to no quantifiable op left.")
        self.re_quant = False

    def _get_op_type_priority(self):
        optypewise_cap = self.capability["optypewise"]
        op_type_priority = list(optypewise_cap.keys())
        return op_type_priority

    def _sorted_item_by_op_type(self, items_lst, op_type_priority: List[str]) -> OrderedDict[str, List]:
        """Scoring the tuning items according to its op type.

        Args:
            items_lst: The tuning item list. # [(op_item, quant_mode), ... ]
            op_type_priority: The op type list with the order. # [optype_1, optype_2]

        Returns:
            The tuning items list that sorted according to its op type.
            OrderDict:
                # op_type: [(TuningItem, quant_mode), ...]
                conv: [(TuningItem, static), (TuningItem, static)]
                linear: [(TuningItem, static), (TuningItem, static)]
                matmul: [(TuningItem, static), (TuningItem, static)]
        """
        sorted_items = COrderedDict()
        for op_item, quant_mode in items_lst:
            op_name, op_type = op_item.name
            for target_op_type in self.quant_op_type_lst:
                # conv: conv1d, conv2d, conv3d
                if op_type.lower().startswith(target_op_type):
                    sorted_items.setdefault(target_op_type, []).append((op_item, quant_mode))
        new_sorted_items = COrderedDict(
            (op_type, sorted_items[op_type]) for op_type in self.quant_op_type_lst if op_type in sorted_items
        )
        return new_sorted_items

    def initialize_tune_cfg(self):
        """Init the tuning config.

        Initialize the tuning config for conservative tuning.

        Returns:
            op_item_dtype_dict (OrderedDict): key is (op_name, op_type); value is quantization mode.
            quant_mode_wise_items (OrderedDict): key is quant_mode/precision; value is item list.
            initial_op_tuning_cfg (OrderedDict): key is (op_name, op_type); value is the initialized tuning config.
        """
        from .utils.constant import auto_query_order_o0 as query_order
        from .utils.tuning_space import initial_tuning_cfg_with_quant_mode

        quant_mode_wise_items = OrderedDict()  # mode, op_item_lst
        pre_items = set()
        # Collect op items supported the specified mode.
        for quant_mode in query_order:
            items = self.tuning_space.query_items_by_quant_mode(quant_mode)
            filtered_items = list(filter(lambda item: item not in pre_items, items))
            pre_items = pre_items.union(set(items))
            quant_mode_wise_items[quant_mode] = filtered_items

        def initial_op_quant_mode(items_lst, target_quant_mode, op_item_dtype_dict):
            for item in items_lst:
                op_item_dtype_dict[item.name] = target_quant_mode

        op_item_dtype_dict = OrderedDict()
        for quant_mode, quant_mode_items in quant_mode_wise_items.items():
            initial_op_quant_mode(quant_mode_items, quant_mode, op_item_dtype_dict)

        initial_op_tuning_cfg = {}
        for op_name_type, quant_mode in op_item_dtype_dict.items():
            initial_op_tuning_cfg[op_name_type] = initial_tuning_cfg_with_quant_mode(
                op_name_type, quant_mode, self.tuning_space
            )
        return op_item_dtype_dict, quant_mode_wise_items, initial_op_tuning_cfg

    def _quant_items_pool(
        self, op_type_priority: List[str]
    ) -> OrderedDict[str, OrderedDict[str, List[Tuple[TuningItem, str]]]]:
        """Create the op queue to be quantized.

        Args:
            op_type_priority: The optype list with priority.

        Returns:
            The op item pool to convert into lower precision.
            quant_items_pool(OrderDict):
                int8:
                    OrderDict:
                        # (TuningItem, quant_mode)
                        conv2d: [(TuningItem, static), (TuningItem, static)]
                        linear: [(TuningItem, static), (TuningItem, static)]
        """
        quant_mode_wise_items = self.tuning_space.quant_mode_wise_items
        # Add all quantized pair into queue
        quant_items_pool = COrderedDict()
        op_item_pairs = []
        quant_ops_name_set = set()
        # collect and sorted all ops that support int8
        for quant_mode, items_lst in quant_mode_wise_items.items():
            if "static" in quant_mode or "dynamic" in quant_mode:
                _quant_mode = "static" if "static" in quant_mode else "dynamic"
                op_item_pairs += [(item, _quant_mode) for item in items_lst if item.name not in quant_ops_name_set]
                quant_ops_name_set = quant_ops_name_set.union([item.name for item in items_lst])
                op_item_pairs = self._sorted_item_by_op_type(op_item_pairs, op_type_priority)
                quant_items_pool["int8"] = op_item_pairs
        return quant_items_pool
