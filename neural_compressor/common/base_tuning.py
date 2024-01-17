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


from typing import Any, Dict, List, Optional, Tuple, Union

from neural_compressor.common.base_config import BaseConfig, ComposableConfig
from neural_compressor.common.logger import Logger

logger = Logger().get_logger()

__all__ = [
    "Evaluator",
    "TuningConfig",
    "Sampler",
    "ConfigLoader",
    "TuningMonitor",
    "TuningLogger",
    "init_tuning",
]


class Evaluator:
    EVAL_FN = "eval_fn"
    WEIGHT = "weight"
    FN_NAME = "name"
    EVAL_FN_TEMPLATE: Dict[str, Any] = {EVAL_FN: None, WEIGHT: 1.0, FN_NAME: None}

    def __init__(self) -> None:
        self.eval_fn_registry: List[Dict[str, Any]] = []

    def evaluate(self, model) -> float:
        """Evaluate the model using registered evaluation functions.

        Args:
            model: The fp32 model or quantized model.

        Returns:
            The overall result of all registered evaluation functions.
        """
        result = 0
        for eval_pair in self.eval_fn_registry:
            eval_fn = eval_pair[self.EVAL_FN]
            eval_result = eval_fn(model)
            result = self._update_the_objective_score(eval_pair, eval_result, result)
        return result

    def _update_the_objective_score(self, eval_pair, eval_result, overall_result) -> float:
        # TODO update the result according to the weight and algo_name
        return overall_result + eval_result * eval_pair[self.WEIGHT]

    def get_number_of_eval_funtions(self) -> int:
        return len(self.eval_fn_registry)

    def _set_eval_fn_registry(self, user_eval_fns: List[Dict]) -> None:
        self.eval_fn_registry = [
            {
                self.EVAL_FN: user_eval_fn_pair[self.EVAL_FN],
                self.WEIGHT: user_eval_fn_pair.get(self.WEIGHT, 1.0),
                self.FN_NAME: user_eval_fn_pair.get(self.FN_NAME, user_eval_fn_pair[self.EVAL_FN].__name__),
            }
            for user_eval_fn_pair in user_eval_fns
        ]

    def set_eval_fn_registry(self, eval_fns: Optional[Union[Dict, List[Dict]]] = None) -> None:
        if eval_fns is None:
            return
        elif isinstance(eval_fns, Dict):
            eval_fns = [eval_fns]
        elif isinstance(eval_fns, List):
            assert all([isinstance(eval_fn_pair, Dict) for eval_fn_pair in eval_fns])
        else:
            raise NotImplementedError(f"The eval_fns should be a dict or a list of dict, but got {type(eval_fns)}.")
        self._set_eval_fn_registry(eval_fns)

    def self_check(self) -> None:
        # check the number of evaluation functions
        num_evaluator = self.get_number_of_eval_funtions()
        assert num_evaluator > 0, "Please ensure that you register at least one evaluation metric for auto-tune."
        logger.info(f"There are {num_evaluator} tuning objectives.")


evaluator = Evaluator()


class TuningConfig:
    """Base Class for Tuning Criterion.

    Args:
        quant_configs: quantization configs. Default value is empty.
        timeout: Tuning timeout (seconds). Default value is 0 which means early stop.
        max_trials: Max tuning times. Default value is 100. Combine with timeout field to decide when to exit.
    """

    def __init__(self, quant_configs=None, timeout=0, max_trials=100, sampler: "Sampler" = None) -> None:
        """Init a TuneCriterion object."""
        self.quant_configs = quant_configs
        self.timeout = timeout
        self.max_trials = max_trials
        self.sampler = sampler


class Sampler:
    pass


class ConfigLoader:
    @staticmethod
    def parse_quant_config(quant_config: BaseConfig) -> List[BaseConfig]:
        if isinstance(quant_config, ComposableConfig):
            result = []
            for q_config in quant_config.config_list:
                result += q_config.expand()
            return result
        else:
            return quant_config.expand()

    def parse_quant_configs(self) -> List[BaseConfig]:
        quant_config_list = []
        for quant_config in self.quant_configs:
            quant_config_list.extend(ConfigLoader.parse_quant_config(quant_config))
        return quant_config_list

    def __init__(self, quant_configs, sampler: Sampler):
        self.quant_configs = quant_configs
        self.sampler = sampler

    def __iter__(self):
        for config in self.parse_quant_configs():
            yield config


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
    """A unified logger for the tuning process.

    It assists validation teams in retrieving logs.
    """

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
    config_loader = ConfigLoader(quant_configs=tuning_config.quant_configs, sampler=tuning_config.sampler)
    tuning_logger = TuningLogger()
    tuning_monitor = TuningMonitor()
    return config_loader, tuning_logger, tuning_monitor
