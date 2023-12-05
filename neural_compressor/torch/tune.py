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


class FakeConfig:
    pass


def get_default_tuning_config():
    return FakeConfig()


from neural_compressor.common.base_tune import Runner, Tuner


class TorchRunner(Runner):
    def __init__(self, model, run_fn, run_args, eval_fn, eval_args) -> None:
        super().__init__()


def autotune(model, tune_config, run_fn, run_args, eval_fn, eval_args):
    tuner = Tuner(tune_config=tune_config)
    runner = TorchRunner(model, run_fn, run_args, eval_fn, eval_args)
    best_qmodel = tuner.search(runner=runner)
    return best_qmodel


from neural_compressor.common.base_tune import BaseTuningConfig


class TuningConfig(BaseTuningConfig):
    def __init__(self, tuning_order, timeout=0, max_trials=100):
        tuning_order = get_default_tuning_config()
        super().__init__(tuning_order, timeout, max_trials)
