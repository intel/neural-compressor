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
"""The HAWQ_V2 tuning strategy."""
from collections import OrderedDict
from copy import deepcopy

from ..utils import logger
from .strategy import TuneStrategy, strategy_registry
from .utils.tuning_sampler import FallbackTuningSampler, OpTypeWiseTuningSampler


@strategy_registry
class HAWQ_V2TuneStrategy(TuneStrategy):
    """The HAWQ V2 tuning strategy.

    HAWQ_V2 implements the "Hawq-v2: Hessian aware trace-weighted quantization of neural networks".
    We made a small change to it by using the hessian trace to score the op impact and then
    fallback the OPs according to the scoring result.
    """

    def next_tune_cfg(self):
        """Generate and yield the next tuning config using HAWQ v2 search in tuning space.

        Returns:
            tune_config (dict): A dict containing the tuning configuration for quantization.
        """
        tuning_space = self.tuning_space
        calib_size = tuning_space.root_item.get_option_by_name("calib_sampling_size").options[0]

        # Initialize the tuning config for each op according to the quantization approach
        op_item_dtype_dict, quant_mode_wise_items, initial_op_tuning_cfg = self.initial_tuning_cfg()
        # Optype-wise tuning tuning items: the algorithm/scheme/granularity of activation(weight)
        early_stop_tuning = True
        stage1_cnt = 0
        quant_ops = quant_mode_wise_items.get("static", [])
        quant_ops += quant_mode_wise_items.get("dynamic", [])
        stage1_max = 1  # TODO set a more appropriate value
        op_wise_tuning_sampler = OpTypeWiseTuningSampler(
            tuning_space, [], [], op_item_dtype_dict, initial_op_tuning_cfg
        )
        for op_tuning_cfg in op_wise_tuning_sampler:
            stage1_cnt += 1
            if early_stop_tuning and stage1_cnt > stage1_max:
                logger.info("Early stopping the stage 1.")
                break
            op_tuning_cfg["calib_sampling_size"] = calib_size
            yield op_tuning_cfg
        # Start compute the hessian trace
        logger.info("**************  Start compute the hessian trace  *****************")
        target_dtype = "fp32"
        hawq_v2_criterion = None
        strategy_kwargs = self.config.tuning_criterion.strategy_kwargs
        if strategy_kwargs:
            hawq_v2_criterion = strategy_kwargs.get("hawq_v2_loss", None)
        # assert hawq_v2_criterion is not None, "HAWQ-V2 strategy needs model loss function to compute the gradient, \
        #     Please assign it by strategy_kwargs({'hawq_v2_loss': hawq_v2_loss})."
        op_to_traces = self.adaptor.calculate_hessian_trace(
            fp32_model=self.model,
            dataloader=self.calib_dataloader,
            q_model=self.last_qmodel,
            criterion=hawq_v2_criterion,
            enable_act=False,
        )
        sorted_op_to_traces = dict(sorted(op_to_traces.items(), key=lambda item: item[1], reverse=True))
        logger.info("**************  Hessian Trace  *****************")
        for op_name, trace in sorted_op_to_traces.items():
            logger.info(f"*** op: {op_name}, hessian trace : {trace}")
        logger.info("************************************************")
        # WA for op mapping
        ordered_ops_tmp = {}
        for op_info in list(initial_op_tuning_cfg.keys()):
            op_name, op_type = op_info
            for op_trace_name in op_to_traces.keys():
                if isinstance(op_trace_name, str) and op_trace_name.startswith(op_name):
                    if op_name in ordered_ops_tmp:
                        logger.info(
                            (
                                f"*** Already assigned the hessian trace to {op_name}",
                                f"update it with the value of {op_trace_name}",
                            )
                        )
                    ordered_ops_tmp[op_name] = op_to_traces[op_trace_name]

        ordered_ops_tmp = sorted(
            ordered_ops_tmp.keys(), key=lambda key: ordered_ops_tmp[key], reverse=self.higher_is_better
        )
        # WA for add op type
        op_info_map = {}
        for op_info in list(initial_op_tuning_cfg.keys()):
            op_info_map[op_info[0]] = op_info  # op_name: (op_name, op_type)
        tmp_ordered_ops = [op_info_map[op_name] for op_name in ordered_ops_tmp]
        op_dtypes = OrderedDict(zip(tmp_ordered_ops, [target_dtype] * len(ordered_ops_tmp)))

        logger.info(f"Start to accumulate fallback to {target_dtype}.")
        initial_op_tuning_cfg = deepcopy(op_tuning_cfg)
        fallback_sampler = FallbackTuningSampler(
            tuning_space,
            tuning_order_lst=[],
            initial_op_tuning_cfg=op_tuning_cfg,
            op_dtypes=op_dtypes,
            accumulate=True,
            skip_first=False,
        )
        for op_tuning_cfg in fallback_sampler:
            op_tuning_cfg["calib_sampling_size"] = calib_size
            yield op_tuning_cfg
