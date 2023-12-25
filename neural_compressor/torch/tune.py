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

from neural_compressor.common.logger import Logger

logger = Logger().get_logger()


class FakeConfig:
    pass


def get_default_tuning_config():
    return FakeConfig()


from neural_compressor.common.base_tune import BaseQuantizer, Tuner


class TorchQuantizer(BaseQuantizer):
    def __init__(self, model, run_fn, run_args) -> None:
        super().__init__()

    def apply(self, quant_config):
        logger.info(f"apply quant_config: {quant_config}.")


def autotune(model, tune_config, run_fn=None, run_args=None):
    tuner = Tuner(tune_config=tune_config)
    runner = TorchQuantizer(model, run_fn, run_args)
    best_qmodel = tuner.search(runner=runner)
    return best_qmodel


from neural_compressor.common.base_tune import BaseTuningConfig


class TuningConfig(BaseTuningConfig):
    def __init__(self, tuning_order=None, timeout=0, max_trials=100):
        if not tuning_order:
            tuning_order = get_default_tuning_config()
        super().__init__(tuning_order, timeout, max_trials)


def get_default_tune_config() -> TuningConfig:
    return TuningConfig()
