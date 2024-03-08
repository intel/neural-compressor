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


import copy
import uuid
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Sized, Tuple, Union

from neural_compressor.common.base_config import BaseConfig
from neural_compressor.common.utils import TuningLogger, logger

__all__ = [
    "Evaluator",
    "EvaluationFuncWrapper",
    "TuningConfig",
    "Sampler",
    "ConfigLoader",
    "TuningMonitor",
    "init_tuning",
    "Sampler",
    "SequentialSampler",
    "default_sampler",
    "ConfigSet",
]


class EvaluationFuncWrapper:
    def __init__(self, eval_fn: Callable, eval_args=None):
        """Evaluation function wrapper.

        Args:
            eval_fn: a function for evaluated the float or quantized model
            eval_args: positional arguments for `eval_fn`
        """
        self.eval_fn = eval_fn
        self.eval_args = eval_args

    def evaluate(self, model) -> Union[float, int]:
        result = self.eval_fn(model, *self.eval_args) if self.eval_args else self.eval_fn(model)
        return result


class Evaluator:
    """Evaluator is a collection of evaluation functions.

    Note: will deprecate this class in the future.

    Examples:
        def eval_acc(model):
            ...

        def eval_perf(molde):
            ...

        # Usage
        user_eval_fns1 = eval_acc
        user_eval_fns2 = {"eval_fn": eval_acc}
        user_eval_fns3 = {"eval_fn": eval_acc, "weight": 1.0, "name": "accuracy"}
        user_eval_fns4 = [
            {"eval_fn": eval_acc, "weight": 0.5},
            {"eval_fn": eval_perf, "weight": 0.5, "name": "accuracy"},
            ]
    """

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
        return overall_result + eval_result * eval_pair[self.WEIGHT]

    def get_number_of_eval_functions(self) -> int:
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

    def set_eval_fn_registry(self, eval_fns: Optional[Union[Callable, Dict, List[Dict]]] = None) -> None:
        # About the eval_fns format, refer the class docstring for details.
        if eval_fns is None:
            return
        elif callable(eval_fns):
            # single eval_fn
            eval_fn_pair = copy.deepcopy(self.EVAL_FN_TEMPLATE)
            eval_fn_pair[self.EVAL_FN] = eval_fns
            eval_fn_pair[self.FN_NAME] = eval_fns.__name__
            eval_fns = [eval_fn_pair]
        elif isinstance(eval_fns, Dict):
            eval_fns = [eval_fns]
        elif isinstance(eval_fns, List):
            assert all([isinstance(eval_fn_pair, Dict) for eval_fn_pair in eval_fns])
        else:
            raise NotImplementedError(f"The eval_fns should be a dict or a list of dict, but got {type(eval_fns)}.")
        self._set_eval_fn_registry(eval_fns)

    def self_check(self) -> None:
        # check the number of evaluation functions
        num_eval_fns = self.get_number_of_eval_functions()
        assert num_eval_fns > 0, "Please ensure that you register at least one evaluation metric for auto-tune."
        logger.info("There are %d evaluations functions.", num_eval_fns)


evaluator = Evaluator()


class ConfigSet:

    def __init__(self, config_list: List[BaseConfig]) -> None:
        self.config_list = config_list

    def __getitem__(self, index) -> BaseConfig:
        assert 0 <= index < len(self.config_list), f"Index {index} out of range."
        return self.config_list[index]

    def __len__(self) -> int:
        return len(self.config_list)

    @classmethod
    def _from_single_config(cls, config: BaseConfig) -> List[BaseConfig]:
        config_list = []
        config_list = config.expand()
        return config_list

    @classmethod
    def _from_list_of_configs(cls, fwk_configs: List[BaseConfig]) -> List[BaseConfig]:
        config_list = []
        for config in fwk_configs:
            config_list += cls._from_single_config(config)
        return config_list

    @classmethod
    def generate_config_list(cls, fwk_configs: Union[BaseConfig, List[BaseConfig]]):
        # There are several cases for the input `fwk_configs`:
        # 1. fwk_configs is a single config
        # 2. fwk_configs is a list of configs
        # For a single config, we need to check if it can be expanded or not.
        config_list = []
        if isinstance(fwk_configs, BaseConfig):
            config_list = cls._from_single_config(fwk_configs)
        elif isinstance(fwk_configs, List):
            config_list = cls._from_list_of_configs(fwk_configs)
        else:
            raise NotImplementedError(f"Unsupported type {type(fwk_configs)} for fwk_configs.")
        return config_list

    @classmethod
    def from_fwk_configs(cls, fwk_configs: Union[BaseConfig, List[BaseConfig]]) -> "ConfigSet":
        """Create a ConfigSet object from a single config or a list of configs.

        Args:
            fwk_configs: A single config or a list of configs.
                Examples:
                    1) single config: RTNConfig(weight_group_size=32)
                    2) single expandable config: RTNConfig(weight_group_size=[32, 64])
                    3) mixed 1) and 2): [RTNConfig(weight_group_size=32), RTNConfig(weight_group_size=[32, 64])]

        Returns:
            ConfigSet: A ConfigSet object.
        """
        config_list = cls.generate_config_list(fwk_configs)
        return cls(config_list)


class Sampler:
    def __init__(self, config_source: Optional[ConfigSet]) -> None:
        pass

    def __iter__(self) -> Iterator[BaseConfig]:
        """Iterate over indices of config set elements."""
        raise NotImplementedError


class SequentialSampler(Sampler):
    """Samples elements sequentially, always in the same order.

    Args:
        config_source (_ConfigSet): config set to sample from
    """

    config_source: Sized

    def __init__(self, config_source: Sized) -> None:
        self.config_source = config_source

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.config_source)))

    def __len__(self) -> int:
        return len(self.config_source)


default_sampler = SequentialSampler


class ConfigLoader:
    def __init__(self, config_set: ConfigSet, sampler: Sampler = default_sampler) -> None:
        self.config_set = ConfigSet.from_fwk_configs(config_set)
        self._sampler = sampler(self.config_set)

    def __iter__(self) -> Generator[BaseConfig, Any, None]:
        for index in self._sampler:
            yield self.config_set[index]


class TuningConfig:
    """Config for auto tuning pipeline.

    Examples:
        from neural_compressor.torch.quantization import TuningConfig
        tune_config = TuningConfig(
            config_set=[config1, config2, ...],
            max_trials=3,
            tolerable_loss=0.01)

        The tuning process stops when either of the following conditions is met:
            1) The number of trials reaches the maximum trials.
            2) The metric loss is within the tolerable loss.

        For condition 2), we calculate the metric loss as follows:
            relative_loss = (fp32_baseline - eval_result_of_q_model) / fp32_baseline
            If relative_loss <= tolerable_loss, we stop the tuning process.
            For example:
                tolerable_loss = 0.01
                fp32_baseline = 100
                eval_result_of_q_model = 99
                relative_loss = (100 - 99) / 100 = 0.01
                The metric loss is within the tolerable loss, so the tuning process is stopped.
    """

    def __init__(
        self,
        config_set: Union[BaseConfig, List[BaseConfig]] = None,
        sampler: Sampler = default_sampler,
        tolerable_loss=0.01,
        max_trials=100,
    ):
        """Initial a TuningConfig.

        Args:
            config_set: A single config or a list of configs. Defaults to None.
            sampler: tuning sampler that decide the trials order. Defaults to default_sampler.
            tolerable_loss: This float indicates how much metric loss we can accept.
                The metric loss is relative, it can be both positive and negative. Default is 0.01.
            max_trials: Max tuning times. Combine with `tolerable_loss` field to decide when to stop. Default is 100.
        """
        self.config_set = config_set
        self.sampler = sampler
        self.tolerable_loss = tolerable_loss
        self.max_trials = max_trials


class _TrialRecord:
    @staticmethod
    def _generate_unique_id():
        unique_id = str(uuid.uuid4())
        return unique_id

    def __init__(self, trial_index: int, trial_result: Union[int, float], quant_config: BaseConfig):
        # The unique id to refer to one trial
        self.trial_id = _TrialRecord._generate_unique_id()
        self.trial_index = trial_index
        self.trial_result = trial_result
        self.quant_config = quant_config


class TuningMonitor:
    def __init__(self, tuning_config: TuningConfig) -> None:
        self.tuning_config = tuning_config
        self.trial_cnt = 0
        self.tuning_history: List[_TrialRecord] = []
        self.baseline = None

    def add_trial_result(self, trial_index: int, trial_result: Union[int, float], quant_config: BaseConfig) -> None:
        self.trial_cnt += 1
        trial_record = _TrialRecord(trial_index, trial_result, quant_config)
        self.tuning_history.append(trial_record)

    def set_baseline(self, baseline: float):
        self.baseline = baseline
        logger.info(f"Fp32 baseline is {self.baseline}")

    def get_number_of_trials(self):
        return len(self.tuning_history)

    def get_best_quant_config(self) -> BaseConfig:
        assert self.get_number_of_trials() > 0, "No trial record in tuning monitor."
        # Put the record with a higher score at the beginning
        sorted_trials_records: List[_TrialRecord] = sorted(
            self.tuning_history, key=lambda x: x.trial_result, reverse=True
        )
        return sorted_trials_records[0].quant_config

    def need_stop(self) -> bool:
        """Check if need to stop tuning. Either accuracy goal is met, max trials is reached or timeout is reached.

        Returns:
            stop_flag: True if need to stop, otherwise False.
        """

        # reach max trials
        reach_max_trials = self.trial_cnt >= self.tuning_config.max_trials
        # reach accuracy goal
        meet_accuracy_goal = (
            False
            if self.baseline is None
            else self.tuning_history[-1].trial_result >= (self.baseline * (1 - self.tuning_config.tolerable_loss))
        )
        # [-1] is the last element representing the latest trail record.
        return reach_max_trials or meet_accuracy_goal


def init_tuning(tuning_config: TuningConfig) -> Tuple[ConfigLoader, TuningLogger, TuningMonitor]:
    config_loader = ConfigLoader(config_set=tuning_config.config_set, sampler=tuning_config.sampler)
    tuning_logger = TuningLogger()
    tuning_monitor = TuningMonitor(tuning_config)
    return config_loader, tuning_logger, tuning_monitor
