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
"""Intel Neural Compressor Tensorflow quantization AutoTune API."""


from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import tensorflow as tf

from neural_compressor.common import logger
from neural_compressor.common.base_config import BaseConfig, get_all_config_set_from_config_registry
from neural_compressor.common.base_tuning import EvaluationFuncWrapper, TuningConfig, init_tuning
from neural_compressor.common.utils import call_counter, dump_elapsed_time
from neural_compressor.tensorflow.quantization import quantize_model
from neural_compressor.tensorflow.quantization.config import FRAMEWORK_NAME, StaticQuantConfig
from neural_compressor.tensorflow.utils import BaseModel, Model, constants

__all__ = [
    "autotune",
    "get_all_config_set",
]


def get_all_config_set() -> Union[BaseConfig, List[BaseConfig]]:
    """Get all config set."""
    return get_all_config_set_from_config_registry(fwk_name=FRAMEWORK_NAME)


@dump_elapsed_time("Pass auto-tune")
@call_counter
def autotune(
    model: Union[str, tf.keras.Model, BaseModel],
    tune_config: TuningConfig,
    eval_fn: Callable,
    eval_args: Optional[Tuple[Any]] = None,
    calib_dataloader: Callable = None,
    calib_iteration: int = 100,
    calib_func: Callable = None,
) -> Optional[BaseModel]:
    """The main entry of auto-tune."""
    model = Model(model)
    best_quant_model = None
    eval_func_wrapper = EvaluationFuncWrapper(eval_fn, eval_args)
    config_loader, tuning_logger, tuning_monitor = init_tuning(tuning_config=tune_config)
    baseline: float = eval_func_wrapper.evaluate(model)
    tuning_monitor.set_baseline(baseline)
    tuning_logger.tuning_start()
    for trial_index, quant_config in enumerate(config_loader, 1):
        tuning_logger.trial_start(trial_index=trial_index)
        tuning_logger.execution_start()
        logger.info(quant_config.to_dict())
        q_model = quantize_model(model, quant_config, calib_dataloader, calib_iteration, calib_func)
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
                del q_model
                best_quant_config: BaseConfig = best_trial_record.quant_config
                best_quant_model = quantize_model(
                    model, best_quant_config, calib_dataloader, calib_iteration, calib_func
                )
            else:
                best_quant_model = q_model
            break
    tuning_logger.tuning_end()
    return best_quant_model
