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
"""The auto-mixed precision strategy."""

import copy
from collections import OrderedDict
from itertools import groupby

from neural_compressor.adaptor.torch_utils.mixed_precision import ipex_mixed_precision

from ..utils import logger
from .strategy import TuneStrategy, strategy_registry
from .utils.tuning_sampler import FallbackTuningSampler
from .utils.tuning_structs import OpTuningConfig


@strategy_registry
class AutoMixedPrecisionTuneStrategy(TuneStrategy):
    """Tuning strategy for auto mixed precision."""

    def _initialize_config(self, conf):
        """Init the tuning config based on user conf.

        Args:
            conf: User config

        Returns:
            Tuning config
        """
        config = conf.mixed_precision
        config.approach = getattr(config, "approach", None)
        config.recipes = getattr(config, "recipes", {})
        config.calibration_sampling_size = getattr(config, "calibration_sampling_size", [0])
        config.op_type_dict = getattr(config, "op_type_dict", None)
        config.op_name_dict = getattr(config, "op_name_dict", None)
        config.quant_format = getattr(config, "quant_format", "")
        config.domain = getattr(config, "domain", None)
        config.reduce_range = getattr(config, "reduce_range", None)
        config.example_inputs = getattr(config, "example_inputs", None)
        config.quant_level = getattr(config, "quant_level", "auto")
        return config

    def next_tune_cfg(self):
        """Generate the next tuning config.

        Tuning configurations are generated according to the following rules:
        1. First, it tries to convert all ops into target date type as many as possible.
        2. If the accuracy does  not meets the requirements, it starts the stage of fallback
            which converts ops into higher precision.

        Yields:
            tune_config (dict): A dict containing the tuning configuration.
        """
        from copy import deepcopy

        # filter quantization dtype
        # TODO align with the old mixed-precison
        target_dtypes = self.config.precisions
        target_dtypes = list(set(target_dtypes) - set(["fp32"]))
        tuning_space = self.tuning_space
        initial_op_tuning_cfg = {}
        for item in tuning_space.root_item.options:
            if item.item_type == "op":
                op_name, op_type = item.name
                initial_op_tuning_cfg[item.name] = OpTuningConfig(op_name, op_type, "fp32", tuning_space)

        if not target_dtypes:
            target_dtypes = ["bf16"]
        # step1. target_dtype AMAP, collect the ops that support target_dtype
        lower_precision_items_name = []
        op_tuning_cfg = {}
        for idx, target_dtype in enumerate(target_dtypes):
            lower_precision_items = tuning_space.query_items_by_quant_mode(target_dtype)
            if len(lower_precision_items) == 0 and not (
                idx == len(target_dtypes) - 1 and len(lower_precision_items_name) == 0
            ):
                continue
            lower_precision_items_name = [item.name for item in lower_precision_items]
            op_tuning_cfg = deepcopy(initial_op_tuning_cfg)
            for op_name_type in lower_precision_items_name:
                op_tuning_cfg[op_name_type] = OpTuningConfig(
                    op_name_type[0], op_name_type[1], target_dtype, tuning_space
                )
            calib_sampling_size = 1
            op_tuning_cfg["calib_sampling_size"] = calib_sampling_size
            yield op_tuning_cfg

        # step 2, fallback op into fp32
        # quant_level:
        #   auto: op-type-wise -> op-wise
        #   0: op-type wise
        #   1: op-wise

        # if quant level is auto or 0, do op type wise fallback
        target_dtype = "fp32"
        fallback_items_name_lst = lower_precision_items_name[::-1]
        if fallback_items_name_lst:
            logger.info("[Strategy] start fallback op into fp32.")
        initial_op_tuning_cfg = deepcopy(op_tuning_cfg)
        if self.config.quant_level in ["auto", 0]:
            logger.info(
                f"[Strategy] fallback op into fp32 in op type wise, \
                as quant level is {self.config.quant_level}"
            )
            for op_tuning_cfg in self.fallback_in_op_type_wise(
                tuning_space, fallback_items_name_lst, deepcopy(initial_op_tuning_cfg), target_dtype
            ):
                yield op_tuning_cfg

        # if quant level is auto or 1, do op instance fallback
        if self.config.quant_level in ["auto", 1]:
            logger.info(
                f"[Strategy] fallback op into fp32 in op wise, \
                as quant level is {self.config.quant_level}"
            )
            for op_tuning_cfg in self.fallback_in_op_wise(
                tuning_space, fallback_items_name_lst, deepcopy(initial_op_tuning_cfg), target_dtype
            ):
                yield op_tuning_cfg

    def fallback_in_op_type_wise(self, tuning_space, fallback_items_name_lst, initial_op_tuning_cfg, target_dtype):
        """Fallback op in op type wise.

        Args:
            tuning_space: tuning space
            fallback_items_name_lst: the list of items to be fallback
            initial_op_tuning_cfg: initial tuning config
            target_dtype: target data type, such as fp32

        Yields:
            tuning config
        """
        fallback_items_name_lst.sort(key=lambda x: x[1])
        op_type_groups = groupby(fallback_items_name_lst, key=lambda x: x[1])
        # key: ((op1_name, op_type1),(op2_name, op_type1), (op3_name, op_type1), ...)
        # value: target dtype
        ops_dtypes = OrderedDict()
        for op_type, op_lst in op_type_groups:
            ops_dtypes[tuple(op_lst)] = target_dtype
        fallback_sampler = FallbackTuningSampler(
            tuning_space,
            tuning_order_lst=[],
            initial_op_tuning_cfg=initial_op_tuning_cfg,
            op_dtypes=ops_dtypes,
            accumulate=False,
        )
        op_fallback_acc_impact = OrderedDict()
        for op_index, op_tuning_cfg in enumerate(fallback_sampler):
            op_tuning_cfg["calib_sampling_size"] = -1
            yield op_tuning_cfg
            acc, _ = self.last_tune_result
            op_fallback_acc_impact[fallback_items_name_lst[op_index]] = acc

    def fallback_in_op_wise(self, tuning_space, fallback_items_name_lst, initial_op_tuning_cfg, target_dtype):
        """Fallback op in op wise.

        Args:
            tuning_space: tuning space
            fallback_items_name_lst: the list of items to be fallback
            initial_op_tuning_cfg: initial tuning config
            target_dtype: target data type, such as fp32

        Yields:
            tuning config
        """
        op_dtypes = OrderedDict(zip(fallback_items_name_lst, [target_dtype] * len(fallback_items_name_lst)))
        fallback_sampler = FallbackTuningSampler(
            tuning_space,
            tuning_order_lst=[],
            initial_op_tuning_cfg=initial_op_tuning_cfg,
            op_dtypes=op_dtypes,
            accumulate=False,
        )
        op_fallback_acc_impact = OrderedDict()
        for op_index, op_tuning_cfg in enumerate(fallback_sampler):
            op_tuning_cfg["calib_sampling_size"] = -1
            yield op_tuning_cfg
            acc, _ = self.last_tune_result
            op_fallback_acc_impact[fallback_items_name_lst[op_index]] = acc

        # do accumulated fallback according to the order in the previous stage
        if len(op_fallback_acc_impact) > 0:
            ordered_ops = sorted(
                op_fallback_acc_impact.keys(),
                key=lambda key: op_fallback_acc_impact[key],
                reverse=self.higher_is_better,
            )
            op_dtypes = OrderedDict(zip(ordered_ops, [target_dtype] * len(fallback_items_name_lst)))
            logger.info("Start to accumulate fallback to {target_dtype}.")
            initial_op_tuning_cfg = copy.deepcopy(op_tuning_cfg)
            fallback_sampler = FallbackTuningSampler(
                tuning_space,
                tuning_order_lst=[],
                initial_op_tuning_cfg=initial_op_tuning_cfg,
                op_dtypes=op_dtypes,
                accumulate=True,
            )
            for op_tuning_cfg in fallback_sampler:
                op_tuning_cfg["calib_sampling_size"] = -1
                yield op_tuning_cfg

    def traverse(self):
        """Traverse the tuning space according to auto-mixed precision strategy."""
        if self.config.backend == "ipex":
            self.best_qmodel = ipex_mixed_precision(self.model, self.config.example_inputs, self.config.device)
            if self.eval_dataloader or self.eval_func:
                self._evaluate(self.best_qmodel)
            return
        self._prepare_tuning()

        for op_tuning_cfg in self.next_tune_cfg():
            # add tune_cfg here as quantize use tune_cfg
            tune_cfg = self._tune_cfg_converter(op_tuning_cfg)
            self.trials_count += 1
            tuning_history = self._find_tuning_history(tune_cfg)
            if tuning_history and self.trials_count < self.config.tuning_criterion.max_trials:
                self.last_tune_result = tuning_history["last_tune_result"]
                self.best_tune_result = tuning_history["best_tune_result"]
                logger.warn("Find evaluated tuning config, skip.")
                continue

            logger.debug("Dump current mixed precision configuration:")
            logger.debug(tune_cfg)
            self.last_qmodel = self.adaptor.quantize(tune_cfg, self.model, self.calib_dataloader, self.q_func)
            assert self.last_qmodel
            # Return the last quantized model as a result. if performance only.
            if self._not_tuning:
                self.best_qmodel = self.last_qmodel
                self._add_tuning_history(copy.deepcopy(tune_cfg), (-1, [0]), q_config=self.last_qmodel.q_config)
                return
            self.last_tune_cfg = copy.deepcopy(tune_cfg)
            if self.eval_dataloader or self.eval_func:
                q_config = copy.deepcopy(self.last_qmodel.q_config)
                self.last_tune_result = self._evaluate(self.last_qmodel)
                self.cur_best_acc, self.cur_best_tuning_cfg = self.update_best_op_tuning_cfg(op_tuning_cfg)
                need_stop = self.stop(self.config.tuning_criterion.timeout, self.trials_count)
                # record the tuning history
                saved_tune_cfg = copy.deepcopy(tune_cfg)
                saved_last_tune_result = copy.deepcopy(self.last_tune_result)
                self._add_tuning_history(saved_tune_cfg, saved_last_tune_result, q_config=q_config)
            else:
                # If the eval_dataloader was not specified under the config yaml file,
                # We only converted the model with customized precisions.
                self.best_qmodel = self.last_qmodel
                need_stop = True

            if need_stop:
                break
