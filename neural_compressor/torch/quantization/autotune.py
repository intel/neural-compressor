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
"""Intel Neural Compressor Pytorch quantization AutoTune API."""


from copy import deepcopy
from typing import Callable, List, Optional, Union

import torch

from neural_compressor.common.base_config import BaseConfig, get_all_config_set_from_config_registry
from neural_compressor.common.base_tuning import EvaluationFuncWrapper, TuningConfig, init_tuning
from neural_compressor.common.utils import dump_elapsed_time
from neural_compressor.torch.quantization import quantize
from neural_compressor.torch.quantization.config import FRAMEWORK_NAME, RTNConfig
from neural_compressor.torch.utils import constants, logger

__all__ = [
    "autotune",
    "get_all_config_set",
    "get_rtn_double_quant_config_set",
]


def get_rtn_double_quant_config_set() -> List[RTNConfig]:
    """Generate RTN double quant config set.

    Returns:
        List[RTNConfig]: a set of quant config
    """
    rtn_double_quant_config_set = []
    for double_quant_type, double_quant_config in constants.DOUBLE_QUANT_CONFIGS.items():
        rtn_double_quant_config_set.append(RTNConfig.from_dict(double_quant_config))
    return rtn_double_quant_config_set


def get_all_config_set() -> Union[BaseConfig, List[BaseConfig]]:
    """Generate all quant config set.

    Returns:
        Union[BaseConfig, List[BaseConfig]]: a set of quant config
    """
    return get_all_config_set_from_config_registry(fwk_name=FRAMEWORK_NAME)


@dump_elapsed_time("Pass auto-tune")
def autotune(
    model: torch.nn.Module,
    tune_config: TuningConfig,
    eval_fn: Callable,
    eval_args=None,
    run_fn=None,
    run_args=None,
    example_inputs=None,
):
    """The main entry of auto-tune.

    Args:
        model (torch.nn.Module): _description_
        tune_config (TuningConfig): _description_
        eval_fn (Callable): for evaluation of quantized models.
        eval_args (tuple, optional): arguments used by eval_fn. Defaults to None.
        run_fn (Callable, optional): for calibration to quantize model. Defaults to None.
        run_args (tuple, optional): arguments used by run_fn. Defaults to None.
        example_inputs (tensor/tuple/dict, optional): used to trace torch model. Defaults to None.

    Returns:
        The quantized model.
    """
    best_quant_model = None
    eval_func_wrapper = EvaluationFuncWrapper(eval_fn, eval_args)
    config_loader, tuning_logger, tuning_monitor = init_tuning(tuning_config=tune_config)
    baseline: float = eval_func_wrapper.evaluate(deepcopy(model))
    tuning_monitor.set_baseline(baseline)
    tuning_logger.tuning_start()
    for trial_index, quant_config in enumerate(config_loader, 1):
        tuning_logger.trial_start(trial_index=trial_index)
        tuning_logger.execution_start()
        logger.info(quant_config.to_dict())
        # !!! Make sure to use deepcopy only when inplace is set to `True`.
        q_model = quantize(
            deepcopy(model),
            quant_config=quant_config,
            run_fn=run_fn,
            run_args=run_args,
            inplace=True,
            example_inputs=example_inputs,
        )
        tuning_logger.execution_end()
        tuning_logger.evaluation_start()
        eval_result: float = eval_func_wrapper.evaluate(q_model)
        tuning_logger.evaluation_end()
        tuning_monitor.add_trial_result(trial_index, eval_result, quant_config)
        tuning_logger.trial_end(trial_index)
        if tuning_monitor.need_stop():
            logger.info("Stopped tuning.")
            best_trial_record = tuning_monitor.get_best_trial_record()
            if best_trial_record.trial_index != trial_index:
                logger.info("Re-quantizing with best quantization config...")
                del q_model  # maybe gc.collect() is needed for memory release
                best_quant_config: BaseConfig = best_trial_record.quant_config
                # !!! Make sure to use deepcopy only when inplace is set to `True`.
                q_model = quantize(
                    deepcopy(model),
                    quant_config=best_quant_config,
                    run_fn=run_fn,
                    run_args=run_args,
                    inplace=True,
                    example_inputs=example_inputs,
                )
            best_quant_model = q_model  # quantize model inplace
            break
    tuning_logger.tuning_end()
    return best_quant_model
