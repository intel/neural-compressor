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
from typing import List, Union


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
        pass

    def get_best_model(self, q_model, eval_res: Union[float, int]):
        pass

    def search(self, runner: Runner):
        for config in self.generate_quant_config():
            q_model = runner.apply(quant_config=config)
            eval_res = runner.evaluate()
            if self.get_best_model(q_model, eval_res):
                return q_model
