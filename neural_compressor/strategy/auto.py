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
"""The auto tuning strategy."""
import copy
from copy import deepcopy
import numpy as np
from collections import OrderedDict
from .strategy import strategy_registry, TuneStrategy, STRATEGIES
from ..utils import logger

from .utils.tuning_sampler import OpTypeWiseTuningSampler, FallbackTuningSampler, ModelWiseTuningSampler
from .utils.tuning_structs import OpTuningConfig
from .utils.constant import TUNING_ITEMS_LST

@strategy_registry
class AutoTuneStrategy(TuneStrategy):
    """The auto tuning strategy.
    
    There are three stages executed by auto strategy sequentially,
    and the tuning process ends once the condition meets the exit policy.
    """
    
    def __init__(self, model, conf, q_dataloader=None, q_func=None, \
        eval_dataloader=None, eval_func=None, resume=None, q_hooks=None):
        super().__init__(model, conf, q_dataloader, q_func, eval_dataloader,\
            eval_func, resume, q_hooks)
        logger.info(f"*** Start auto tuning")
        self.model = model
        self.conf = conf
        self.q_dataloader = q_dataloader
        self.q_func = q_func
        self.eval_dataloader = eval_dataloader
        self.eval_func = eval_func
        self.resume = resume
        self.q_hooks = q_hooks
        self.strategies_sequence = ['fwkquant','conservative', 'basic']


    def traverse(self):
        """Generate and yield the next tuning config.

        Returns:
            tune_config (dict): A dict containing the tuning configuration for quantization.
        """
        pre_strategy = None
        for strategy_name in self.strategies_sequence:
            logger.info(f"*** Start {strategy_name} tuning.")
            strategy = STRATEGIES[strategy_name](self.model, self.conf, self.q_dataloader, self.q_func, \
                self.eval_dataloader, self.eval_func, self.resume, self.q_hooks)
            if pre_strategy:
                #TODO add tuning history from the previous stage to current stage.
                strategy.baseline = deepcopy(pre_strategy.baseline)
                strategy.trials_count = pre_strategy.trials_count
                strategy.objectives.baseline = deepcopy(pre_strategy.baseline)
            pre_strategy = strategy
            strategy.traverse()
            self.best_qmodel = strategy.best_qmodel
            if self.best_qmodel:
                return 