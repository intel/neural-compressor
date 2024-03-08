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

from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import tensorflow as tf

from neural_compressor.common import logger
from neural_compressor.common.base_config import BaseConfig, get_all_config_set_from_config_registry
from neural_compressor.common.base_tuning import EvaluationFuncWrapper, TuningConfig, init_tuning
from neural_compressor.common.utils import dump_elapsed_time
from neural_compressor.tensorflow.quantization import quantize_model
from neural_compressor.tensorflow.quantization.config import FRAMEWORK_NAME, StaticQuantConfig
from neural_compressor.tensorflow.utils import BaseModel, constants

__all__ = [
    "autotune",
    "get_all_config_set",
]


def get_all_config_set() -> Union[BaseConfig, List[BaseConfig]]:
    return get_all_config_set_from_config_registry(fwk_name=FRAMEWORK_NAME)


@dump_elapsed_time("Pass auto-tune")
def autotune(
    model: Union[str, tf.keras.Model, BaseModel],
    tune_config: TuningConfig,
    eval_fn: Callable,
    eval_args: Optional[Tuple[Any]] = None,
    calib_dataloader: Callable = None,
    calib_iteration: int = 100,
) -> Optional[BaseModel]:
    """The main entry of auto-tune."""
    best_quant_model = None
    eval_func_wrapper = EvaluationFuncWrapper(eval_fn, eval_args)
    config_loader, tuning_logger, tuning_monitor = init_tuning(tuning_config=tune_config)
    baseline: float = eval_func_wrapper.evaluate(model)
    tuning_monitor.set_baseline(baseline)
    tuning_logger.tuning_start()
    for trial_index, quant_config in enumerate(config_loader):
        tuning_logger.trial_start(trial_index=trial_index)
        tuning_logger.quantization_start()
        logger.info(quant_config.to_dict())
        q_model = quantize_model(model, quant_config, calib_dataloader, calib_iteration)
        tuning_logger.quantization_end()
        tuning_logger.evaluation_start()
        eval_result: float = eval_func_wrapper.evaluate(q_model)
        tuning_logger.evaluation_end()
        tuning_monitor.add_trial_result(trial_index, eval_result, quant_config)
        tuning_logger.trial_end(trial_index)
        if tuning_monitor.need_stop():
            logger.info("Stopped tuning.")
            best_quant_config: BaseConfig = tuning_monitor.get_best_quant_config()
            best_quant_model = quantize_model(model, quant_config, calib_dataloader, calib_iteration)
            break
    tuning_logger.tuning_end()
    return best_quant_model
