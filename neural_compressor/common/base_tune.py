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

from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

from neural_compressor.common.base_config import BaseConfig, ComposableConfig
from neural_compressor.common.logger import Logger

logger = Logger().get_logger()


class FrameworkWrapper:
    """Abstract base class for wrap framework's APIs.

    FrameworkWrapper provides a uniform interface for encapsulating different framework's APIs.
    This class is intended to be used by a `tuner` to obtain quantized models.
    """

    def __init__(self, model) -> None:
        self.model = model

    @abstractmethod
    def apply(self) -> Any:
        """The entry to apply algorithms on a given model."""
        raise NotImplementedError


class TuningObjectives:
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

    def get_number_of_tuning_objectives(self) -> int:
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


tuning_objectives = TuningObjectives()


class BaseTuningConfig:
    """Base Class for Tuning Criterion.

    Args:
        quant_configs: quantization configs. Default value is empty.
        timeout: Tuning timeout (seconds). Default value is 0 which means early stop.
        max_trials: Max tune times. Default value is 100. Combine with timeout field to decide when to exit.
    """

    def __init__(self, quant_configs=None, timeout=0, max_trials=100) -> None:
        """Init a TuningCriterion object."""
        self.quant_configs = quant_configs
        self.timeout = timeout
        self.max_trials = max_trials


class Tuner:
    def __init__(
        self, tune_config: BaseTuningConfig, tuning_objectives: TuningObjectives, fwk_wrapper: FrameworkWrapper
    ) -> None:
        self.tune_config = tune_config
        self.tuning_objectives = tuning_objectives
        self.fwk_wrapper = fwk_wrapper
        self._post_init()

    def _post_init(self) -> None:
        # check the number of evaluation functions
        num_tuning_objectives = self.tuning_objectives.get_number_of_tuning_objectives()
        assert (
            num_tuning_objectives > 0
        ), "Please ensure that you register at least one evaluation metric for auto-tune."
        logger.info(f"There are {num_tuning_objectives} tuning objectives.")

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
        for quant_config in self.tune_config.quant_configs:
            quant_config_list.extend(Tuner.parse_quant_config(quant_config))
        return quant_config_list

    def get_best_model(self, q_model, objective_score: Union[float, int]) -> Any:
        # TODO(Yi) enable it at the next PR
        pass

    def get_tuning_objective_score(self, model) -> float:
        eval_result = self.tuning_objectives.evaluate(model)
        return eval_result

    def search(self) -> Any:
        for config in self.parse_quant_configs():
            logger.info(f"config {config}")
            q_model = self.fwk_wrapper.apply(quant_config=config)
            if self.get_best_model(q_model, self.get_tuning_objective_score(q_model)):
                return q_model
