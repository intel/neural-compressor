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

from typing import Callable, Dict, List, Optional, Tuple, Union

import torch

from neural_compressor.common.base_tune import BaseTuningConfig, FrameworkWrapper, Tuner, tuning_objectives
from neural_compressor.common.logger import Logger
from neural_compressor.torch.quantization.config import GPTQConfig, RTNWeightQuantConfig

logger = Logger().get_logger()


def get_default_tuning_config():
    # TODO (Yi) support it in the next PR
    return None


class TorchWrapper(FrameworkWrapper):
    """Concrete implementation of `FrameworkWrapper` for PyTorch models."""

    def __init__(
        self, model: torch.nn.Module, run_fn: Optional[Callable] = None, run_args: Optional[Tuple] = None
    ) -> None:
        super().__init__(model)
        self.run_fn = run_fn
        self.run_args = run_args

    def apply(self, quant_config):
        """The entry to apply quantization algorithms on a given a model."""
        logger.info(f"apply quant_config: {quant_config}.")
        from neural_compressor.torch import quantize

        q_model = quantize(model=self.model, quant_config=quant_config, run_fn=self.run_fn, run_args=self.run_args)
        return q_model


class TuningConfig(BaseTuningConfig):
    def __init__(self, quant_configs=None, timeout=0, max_trials=100):
        super().__init__(quant_configs, timeout, max_trials)


def autotune(
    model: torch.nn.Module,
    tune_config: TuningConfig,
    eval_fns: Optional[Union[Dict, List[Dict]]] = None,
    run_fn=None,
    run_args=None,
):
    tuning_objectives.set_eval_fn_registry(eval_fns)
    torch_wrapper = TorchWrapper(model, run_fn, run_args)
    tuner = Tuner(tune_config=tune_config, tuning_objectives=tuning_objectives, fwk_wrapper=torch_wrapper)
    best_qmodel = tuner.search()
    return best_qmodel


def get_default_tune_config():
    # TODO use the registered default tuning config in the next PR
    return TuningConfig(quant_configs=[GPTQConfig(weight_bits=[4, 8]), RTNWeightQuantConfig(weight_bits=[4, 8])])
