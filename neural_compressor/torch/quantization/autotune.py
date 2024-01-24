# Copyright (c) 2023 Intel Corporation
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

from copy import deepcopy
from typing import Dict, List, Optional, Union

import torch

from neural_compressor.common import Logger
from neural_compressor.common.base_config import BaseConfig, get_all_config_set_from_config_registry
from neural_compressor.common.base_tuning import TuningConfig, evaluator, init_tuning
from neural_compressor.torch import quantize
from neural_compressor.torch.quantization.config import FRAMEWORK_NAME

logger = Logger().get_logger()


__all__ = [
    "autotune",
    "get_all_config_set",
]


def get_all_config_set() -> Union[BaseConfig, List[BaseConfig]]:
    return get_all_config_set_from_config_registry(fwk_name=FRAMEWORK_NAME)


def autotune(
    model: torch.nn.Module,
    tune_config: TuningConfig,
    eval_fns: Optional[Union[Dict, List[Dict]]] = None,
    run_fn=None,
    run_args=None,
) -> Optional[torch.nn.Module]:
    """The main entry of auto-tune."""
    best_quant_model = None
    evaluator.set_eval_fn_registry(eval_fns)
    evaluator.self_check()
    config_loader, tuning_logger, tuning_monitor = init_tuning(tuning_config=tune_config)
    tuning_logger.tuning_start()
    for trial_index, quant_config in enumerate(config_loader):
        tuning_logger.trial_start(trial_index=trial_index)
        tuning_logger.quantization_start()
        logger.info(f"quant config: {quant_config}")
        # !!! Make sure to use deepcopy only when inplace is set to `True`.
        q_model = quantize(deepcopy(model), quant_config=quant_config, run_fn=run_fn, run_args=run_args, inplace=True)
        tuning_logger.quantization_end()
        tuning_logger.evaluation_start()
        eval_result: float = evaluator.evaluate(q_model)
        tuning_logger.evaluation_end()
        tuning_monitor.add_trial_result(trial_index, eval_result, quant_config)
        if tuning_monitor.need_stop():
            best_quant_config: BaseConfig = tuning_monitor.get_best_quant_config()
            # !!! Make sure to use deepcopy only when inplace is set to `True`.
            quantize(deepcopy(model), quant_config=best_quant_config, run_fn=run_fn, run_args=run_args, inplace=True)
            best_quant_model = model  # quantize model inplace
        tuning_logger.trial_end(trial_index)
    tuning_logger.tuning_end()
    return best_quant_model
