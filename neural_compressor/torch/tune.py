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

from typing import Callable, Dict, Generator, List, Optional, Tuple, Union

import torch

from neural_compressor.common.base_config import BaseConfig
from neural_compressor.common.base_tune import TuningConfig, evaluator
from neural_compressor.common.logger import Logger
from neural_compressor.torch import quantize
from neural_compressor.torch.quantization.config import GPTQConfig, RTNWeightQuantConfig

logger = Logger().get_logger()


class Sampler:
    pass


class ConfigLoader:
    def __iter__(self) -> Generator[BaseConfig]:
        yield None


class TuningMonitor:
    def __init__(self) -> None:
        # TODO refine the `tuning_history` with a more appropriate data structure
        self.tuning_history: list = []

    def add_trial_result(self, trial_index: int, eval_result: Union[int, float], quant_config: BaseConfig) -> None:
        self.tuning_history.append([trial_index, eval_result, quant_config])

    def get_best_quant_config(self) -> BaseConfig:
        return self.tuning_history[0][2]

    def need_stop(self) -> bool:
        return True


class TuningLogger:
    @classmethod
    def tuning_start(cls) -> None:
        logger.info("Tuning started.")

    @classmethod
    def trial_start(cls, trial_index: int = None) -> None:
        logger.info(f" {trial_index}-trail started.")

    @classmethod
    def quantization_start(cls) -> None:
        logger.info("Quantization started.")

    @classmethod
    def quantization_end(cls) -> None:
        logger.info("Quantization end.")

    @classmethod
    def evaluation_start(cls) -> None:
        logger.info("Evaluation started.")

    @classmethod
    def evaluation_end(cls) -> None:
        logger.info("Evaluation end.")

    @classmethod
    def trial_end(cls, trial_index: int = None) -> None:
        logger.info(f" {trial_index}-trail end.")

    @classmethod
    def tuning_end(cls) -> None:
        logger.info("Tuning completed.")


def init_tuning(tuning_config: TuningConfig) -> Tuple[ConfigLoader, TuningLogger, TuningMonitor]:
    config_loader = ConfigLoader()
    tuning_logger = TuningLogger()
    tuning_monitor = TuningMonitor()
    return config_loader, tuning_logger, tuning_monitor


def get_default_tune_config():
    # TODO use the registered default tune config in the next PR
    return TuningConfig(quant_configs=[GPTQConfig(weight_bits=[4, 8]), RTNWeightQuantConfig(weight_bits=[4, 8])])


def autotune(
    model: torch.nn.Module,
    tune_config: TuningConfig,
    eval_fns: Optional[Union[Dict, List[Dict]]] = None,
    run_fn=None,
    run_args=None,
) -> Optional[torch.nn.Module]:
    # TODO Old Impl, remove it before merge
    # evaluator.set_eval_fn_registry(eval_fns)
    # torch_wrapper = TorchWrapper(model, run_fn, run_args)
    # tuner = Tuner(float_model=model, tune_config=tune_config, evaluator=evaluator, fwk_wrapper=torch_wrapper)
    # best_qmodel = tuner.search()

    best_quant_model = None
    evaluator.set_eval_fn_registry(eval_fns)
    config_loader, tuning_logger, tuning_monitor = init_tuning(tuning_config=tune_config)
    tuning_logger.tuning_start()
    for trial_index, quant_config in enumerate(config_loader):
        tuning_logger.trial_start(trial_index=trial_index)
        tuning_logger.quantization_start()
        q_model = quantize(model, quant_config=quant_config, run_fn=run_fn, run_args=run_args)
        tuning_logger.quantization_end()
        tuning_logger.evaluation_start()
        eval_result: float = evaluator.evaluate(q_model)
        tuning_logger.evaluation_end()
        tuning_monitor.add_trial_result(trial_index, eval_result, quant_config)
        if tuning_monitor.need_stop():
            best_quant_config: BaseConfig = tuning_monitor.get_best_quant_config()
            best_quant_model = quantize(model, quant_config=best_quant_config, run_fn=run_fn, run_args=run_args)()
        tuning_logger.trial_end()
    tuning_logger.tuning_end()
    return best_quant_model
