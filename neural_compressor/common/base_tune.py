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
from typing import Any, Callable, List, Union

from neural_compressor.common.logger import Logger

logger = Logger().get_logger()


class Runner:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def apply(self, quant_config):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError


class BaseTuningConfig:
    """Base Class for Tuning Criterion.

    Args:
        tuning_order: the traverse order.
        timeout: Tuning timeout (seconds). Default value is 0 which means early stop.
        max_trials: Max tune times. Default value is 100. Combine with timeout field to decide when to exit.
    """

    def __init__(self, tuning_order=None, timeout=0, max_trials=100):
        """Init a TuningCriterion object."""
        self.tuning_order = tuning_order
        self.timeout = timeout
        self.max_trials = max_trials


class Tuner:
    def __init__(self, tune_config: BaseTuningConfig):
        self.tune_config = tune_config

    def generate_quant_config(self):
        return [None]

    def get_best_model(self, q_model, objective_score: Union[float, int]):
        pass

    def get_objective_score(self, model):
        eval_result = objective.evaluate(model)
        return eval_result

    def search(self, runner: Runner):
        for config in self.generate_quant_config():
            q_model = runner.apply(quant_config=config)
            if self.get_best_model(q_model, self.get_objective_score(q_model)):
                return q_model


class Objective:
    def __init__(self) -> None:
        self.eval_fn_registry = []

    def evaluate(self, model):
        result = 0
        for eval_pair in self.eval_fn_registry:
            eval_fn = eval_pair["func"]
            result += eval_fn(model)
        return result


objective = Objective()


def register_tuning_target(algo_name="rtn_weight_only_quant", weight=0.5, mode="max", name=None) -> Callable[..., Any]:
    def decorator(eval_fn):
        fn_name = name if name else eval_fn.__name__
        eval_pair = {"name": fn_name, "algo_name": algo_name, "weight": weight, "mode": mode, "func": eval_fn}
        objective.eval_fn_registry.append(eval_pair)
        logger.info(
            f"Add new tuning target : {eval_pair} to tuning target registry({len(objective.eval_fn_registry)})."
        )
        return eval_fn

    return decorator
