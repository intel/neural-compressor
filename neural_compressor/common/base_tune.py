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


class AlgorithmManager:
    """Abstract base class to manage the cross-framework algorithms.

    This class is designed to be used by a tuner to obtain a quantized model.
    """

    def __init__(self, model) -> None:
        self.model = model

    @abstractmethod
    def apply(self):
        """The entry to apply algorithms on a given model."""
        raise NotImplementedError


class TuningObjective:
    def __init__(self) -> None:
        self.eval_fn_registry: List[Callable] = []

    def evaluate(self, model) -> float:
        """Evaluate the model using registered evaluation functions.

        Args:
            model: The fp32 model or quantized model.

        Returns:
            The overall result of all registered evaluation functions.
        """
        result = 0
        for eval_pair in self.eval_fn_registry:
            # eval_pair: {"name": fn_name, "algo_name": algo_name, "weight": weight, "func": eval_fn}
            eval_fn = eval_pair["func"]
            eval_result = eval_fn(model)
            result = self._update_the_objective_score(eval_pair, eval_result, result)
        return result

    def _update_the_objective_score(self, eval_pair, eval_result, overall_result):
        # TODO update the result according to the weight and algo_name
        return overall_result + eval_result * eval_pair["weight"]

    def get_number_of_tuning_objectives(self):
        return len(self.eval_fn_registry)

    def _update_eval_fn_registry(self, eval_fns: List[Dict]):
        self.eval_fn_registry = eval_fns

    def update_eval_fn_registry(self, eval_fns: Optional[Union[Dict, List[Dict]]] = None) -> None:
        if eval_fns is None:
            return
        elif isinstance(eval_fns, Dict):
            eval_fns = [eval_fns]
        elif isinstance(eval_fns, List):
            assert all([isinstance(eval_fn_pair, Dict) for eval_fn_pair in eval_fns])
        else:
            raise NotImplementedError(f"The eval_fns should be a dict or a list of dict, but got {type(eval_fns)}.")
        self._update_eval_fn_registry(eval_fns)


tuning_objective = TuningObjective()


def register_tuning_objective(
    algo_name: Optional[str] = None, weight: float = 0.5, objective_name: Optional[str] = None
) -> Callable[..., Any]:
    """Decorator for registering a tuning objective evaluation function.

    Args:
        algo_name (Optional[str]): Algorithm name associated with the tuning objective.
            The default value is None, which applies to all algorithms.
        weight (float): Weight assigned to the tuning objective.
        objective_name (Optional[str]): Optional name for the tuning objective. If no name is provided,
            the function's built-in name will be used instead.

    Returns:
        Callable[..., Any]: Decorator function for the evaluation function.

    Usage:
        @register_tuning_objective(algo_name="rtn_weight_only_quant", weight=0.7, objective_name="eval_accuracy")
        def eval_accuracy(model):
            # Evaluate the accuracy of the given model.
            return score
    """

    def decorator(eval_fn: Callable) -> Callable:
        fn_name = objective_name if objective_name else eval_fn.__name__
        eval_pair = {"name": fn_name, "algo_name": algo_name, "weight": weight, "func": eval_fn}
        tuning_objective.eval_fn_registry.append(eval_pair)
        logger.info(
            f"Add new tuning objective : {eval_pair} to tuning objective registry("
            f"{tuning_objective.get_number_of_tuning_objectives()})."
        )
        return eval_fn

    return decorator


class BaseTuningConfig:
    """Base Class for Tuning Criterion.

    Args:
        quant_configs: quantization configs. Default value is empty.
        timeout: Tuning timeout (seconds). Default value is 0 which means early stop.
        max_trials: Max tune times. Default value is 100. Combine with timeout field to decide when to exit.
    """

    def __init__(self, quant_configs=None, timeout=0, max_trials=100):
        """Init a TuningCriterion object."""
        self.quant_configs = quant_configs
        self.timeout = timeout
        self.max_trials = max_trials


class Tuner:
    def __init__(self, tune_config: BaseTuningConfig):
        self.tune_config = tune_config
        self.tuner_tuning_objective = tuning_objective
        self._post_init()

    def _post_init(self):
        # check the number of evaluation functions
        num_tuning_objectives = self.tuner_tuning_objective.get_number_of_tuning_objectives()
        assert (
            num_tuning_objectives > 0
        ), "Please ensure that you register at least one evaluation metric for auto-tune."
        logger.info(f"There are {num_tuning_objectives} tuning objectives.")

    @staticmethod
    def generate_quant_config(quant_config: BaseConfig) -> List[BaseConfig]:
        if isinstance(quant_config, ComposableConfig):
            result = []
            for q_config in quant_config.config_list:
                result += q_config.expand()
            return result
        else:
            return quant_config.expand()

    def generate_quant_config_from_quant_configs(self):
        quant_config_list = []
        for quant_config in self.tune_config.quant_configs:
            quant_config_list.extend(Tuner.generate_quant_config(quant_config))
        return quant_config_list

    def get_best_model(self, q_model, objective_score: Union[float, int]):
        pass

    def get_tuning_objective_score(self, model):
        eval_result = self.tuner_tuning_objective.evaluate(model)
        return eval_result

    def search(self, algo_manager: AlgorithmManager):
        for config in self.generate_quant_config_from_quant_configs():
            logger.info(f"config {config}")
            q_model = algo_manager.apply(quant_config=config)
            if self.get_best_model(q_model, self.get_tuning_objective_score(q_model)):
                return q_model
