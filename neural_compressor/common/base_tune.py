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
from typing import Any, Callable, List, Optional, Union

from neural_compressor.common.base_config import BaseConfig, ComposableConfig
from neural_compressor.common.logger import Logger

logger = Logger().get_logger()


class BaseQuantizer:
    """Abstract base class representing a cross-framework quantizer.

    This class is designed to be used by a tuner to obtain a quantized model.
    """

    def __init__(self, model) -> None:
        self.model = model

    @abstractmethod
    def prepare(self, quant_config):
        """Prepare a copy of the model for quantization."""
        raise NotImplementedError

    @abstractmethod
    def calibrate(self):
        """Run the prepared model on the calibration dataset."""
        raise NotImplementedError

    @abstractmethod
    def convert(self):
        "Convert a calibrated model to quantized model."
        raise NotImplementedError

    @abstractmethod
    def quantize(self):
        """The entry to quantize a model."""
        raise NotImplementedError


class TargetManager:
    def __init__(self) -> None:
        self.eval_fn_registry: List[Callable] = []

    def evaluate(self, model, algo_name: Optional[str] = None) -> float:
        """Evaluate the model using registered evaluation functions.

        Args:
            model: The fp32 model or quantized model.
            algo_name: _description_. Defaults to None.

        Returns:
            _description_
        """
        result = 0
        for eval_pair in self.eval_fn_registry:
            # eval_pair: {"name": fn_name, "algo_name": algo_name, "weight": weight, "func": eval_fn}
            eval_fn = eval_pair["func"]
            eval_result = eval_fn(model)
            result = self._update_the_target_score(eval_pair, eval_result, result)
        return result

    def _update_the_target_score(self, eval_pair, eval_result, overall_result):
        # TODO update the result according to the weight and algo_name
        return overall_result + eval_result * eval_pair["weight"]

    def get_number_of_tuning_targets(self):
        return len(self.eval_fn_registry)


target_manager = TargetManager()


def register_tuning_target(
    algo_name: Optional[str] = None, weight: float = 0.5, target_name: Optional[str] = None
) -> Callable[..., Any]:
    """Decorator for registering a tuning target evaluation function.

    Args:
        algo_name (Optional[str]): Algorithm name associated with the tuning target.
            The default value is None, which applies to all algorithms.
        weight (float): Weight assigned to the tuning target.
        target_name (Optional[str]): Optional name for the tuning target. If no name is provided,
            the function's built-in name will be used instead.

    Returns:
        Callable[..., Any]: Decorator function for the evaluation function.

    Usage:
        @register_tuning_target(algo_name="rtn_weight_only_quant", weight=0.7, target_name="eval_accuracy")
        def eval_accuracy(model):
            # Evaluate the accuracy of the given model.
            return score
    """

    def decorator(eval_fn: Callable) -> Callable:
        fn_name = target_name if target_name else eval_fn.__name__
        eval_pair = {"name": fn_name, "algo_name": algo_name, "weight": weight, "func": eval_fn}
        target_manager.eval_fn_registry.append(eval_pair)
        logger.info(
            f"Add new tuning target : {eval_pair} to tuning target registry({len(target_manager.eval_fn_registry)})."
        )
        return eval_fn

    return decorator


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
        self.tuner_target_manager = target_manager
        self._post_init()

    def _post_init(self):
        # check the number of evaluation functions
        num_tuning_targets = self.tuner_target_manager.get_number_of_tuning_targets()
        assert num_tuning_targets > 0, "Please ensure that you register at least one evaluation metric for auto-tune."
        logger.info(f"There are {num_tuning_targets} tuning targets.")

    @staticmethod
    def generate_quant_config(quant_config: BaseConfig) -> List[BaseConfig]:
        if isinstance(quant_config, ComposableConfig):
            result = []
            for q_config in quant_config.config_list:
                result += q_config.expand()
            return result
        else:
            return quant_config.expand()

    def generate_quant_config_from_tuning_order(self):
        quant_config_list = []
        for quant_config in self.tune_config.tuning_order:
            quant_config_list.extend(Tuner.generate_quant_config(quant_config))
        return quant_config_list

    def get_best_model(self, q_model, objective_score: Union[float, int]):
        pass

    def get_tuning_target_score(self, model):
        eval_result = self.tuner_target_manager.evaluate(model)
        return eval_result

    def search(self, quantizer: BaseQuantizer):
        for config in self.generate_quant_config_from_tuning_order():
            logger.info(f"config {config}")
            q_model = quantizer.quantize(quant_config=config)
            if self.get_best_model(q_model, self.get_tuning_target_score(q_model)):
                return q_model
