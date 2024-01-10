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
from typing import Any, Dict, List, Optional, Union

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


class TuneObjectives:
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

    def get_number_of_tune_objectives(self) -> int:
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


tune_objectives = TuneObjectives()


class BaseTuneConfig:
    """Base Class for Tune Criterion.

    Args:
        quant_configs: quantization configs. Default value is empty.
        timeout: Tune timeout (seconds). Default value is 0 which means early stop.
        max_trials: Max tune times. Default value is 100. Combine with timeout field to decide when to exit.
    """

    def __init__(self, quant_configs=None, timeout=0, max_trials=100) -> None:
        """Init a TuneCriterion object."""
        self.quant_configs = quant_configs
        self.timeout = timeout
        self.max_trials = max_trials


class Trial:
    def __init__(
        self, float_model, quant_config: BaseConfig, fwk_wrapper: FrameworkWrapper, tune_objectives: TuneObjectives
    ):
        # The unique id to refer to one trial, it's used by the tuner.
        self.trial_id = None
        self._trial_result = None
        self.set_trail_result_cnt = 0
        self.float_model = float_model
        self.quant_model = None
        self.quant_config = quant_config
        self.fwk_wrapper = fwk_wrapper
        self.tune_objectives = tune_objectives
        self._post_init()

    def _post_init(self):
        """Post initialize one trial."""
        # generate the trial_id
        pass

    @property
    def trial_result(self):
        return self._trial_result

    @trial_result.setter
    def trial_result(self, result):
        assert self.set_trail_result_cnt < 1, "The trial result already be set."
        self._trial_result = result
        self.set_trail_result_cnt += 1

    def quantize(self):
        """Quantize the model with given quant_config."""
        quant_model = self.fwk_wrapper.apply(self.quant_config)
        self.quant_model = quant_model
        return quant_model

    def get_eval_result(self) -> float:
        """Retune the evaluation result.

        The evaluation process is triggered by Lazy only when it is needed, and it is called only once.
        """
        if not self.trial_result:
            eval_score = self.tune_objectives.evaluate(self.quant_model)
            self.trial_result = eval_score
        return self.trial_result

    def recover_quant_model(self):
        """The quantized model should be destroyed after evaluation to save the memory
        and recovery it before end the tuning process."""
        pass

    def destroy_quant_model(self) -> None:
        """"""
        pass


class Tuner:
    def __init__(
        self, float_model, tune_config: BaseTuneConfig, tune_objectives: TuneObjectives, fwk_wrapper: FrameworkWrapper
    ) -> None:
        self.float_model = float_model
        self.tune_config = tune_config
        self.tune_objectives = tune_objectives
        self.fwk_wrapper = fwk_wrapper
        self._post_init()

    def _post_init(self) -> None:
        # check the number of evaluation functions
        num_tune_objectives = self.tune_objectives.get_number_of_tune_objectives()
        assert num_tune_objectives > 0, "Please ensure that you register at least one evaluation metric for auto-tune."
        logger.info(f"There are {num_tune_objectives} tune objectives.")

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

    def get_best_model(self) -> Any:
        # TODO(Yi) enable it at the next PR
        pass

    def needs_stop(self):
        return False

    def update_tune_history(self, trial: Trial):
        pass

    def search(self) -> Any:
        for config in self.parse_quant_configs():
            logger.info(f"Config {config}")
            trial = Trial(self.float_model, config, fwk_wrapper=self.fwk_wrapper, tune_objectives=self.tune_objectives)
            trial.quantize()
            trial.get_eval_result()
            self.update_tune_history(trial)
            if self.needs_stop():
                return self.get_best_model()
